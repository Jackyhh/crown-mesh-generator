#!/usr/bin/env python3
"""
Crown Inference Script - No Visualization

This script performs inference on the trained crown generation model and saves
the generated meshes and point clouds to disk for later download and viewing.
No visualization or display is performed - only data saving.

Usage:
    python infer_crowns.py [--config CONFIG] [--model MODEL] [--output_dir OUTPUT_DIR]
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import shutil, argparse, time, os
import pandas as pd
from collections import defaultdict
from pathlib import Path
from src import config
from src.utils import mc_from_psr, export_mesh, export_pointcloud
from src.dpsr import DPSR
from src.training import Trainer
from src.model import Encode2Points
from src.utils import load_config, load_model_manual, scale2onet, is_url, load_url
from tqdm import tqdm
import json
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Crown Inference - No Visualization')
    parser.add_argument('--config', type=str, default='configs/crown_training.yaml',
                        help='Path to config file.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--output_dir', type=str, default='inference_output',
                        help='Output directory for generated data')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')    
    parser.add_argument('--seed', type=int, default=1, metavar='S', 
                        help='random seed (default: 1)')
    parser.add_argument('--split', type=str, default='test', 
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use for inference')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Maximum number of samples to process (None = all)')
    
    args = parser.parse_args()
    
    print("Crown Inference Script - Data Generation Only")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max samples: {args.num_samples if args.num_samples else 'All'}")
    
    # Load config
    try:
        cfg = load_config(args.config, 'configs/default.yaml')
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return 1
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"crown_inference_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to: {output_root}")
    
    # Setup output directories
    mesh_dir = output_root / "meshes"
    pointcloud_dir = output_root / "pointclouds"
    input_dir = output_root / "inputs"
    metadata_dir = output_root / "metadata"
    
    for dir_path in [mesh_dir, pointcloud_dir, input_dir, metadata_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Load dataset
    try:
        dataset = config.get_dataset(args.split, cfg, return_idx=True)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=0, shuffle=False)
        print(f"Loaded {len(dataset)} samples from {args.split} split")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return 1
    
    # Load model
    model = Encode2Points(cfg).to(device)
    
    # Determine model path
    out_dir = cfg['train']['out_dir']
    if args.model:
        model_path = args.model
    else:
        model_path = os.path.join(out_dir, 'model_best.pt')
    
    try:
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            return 1
            
        print(f"Loading model: {model_path}")
        state_dict = torch.load(model_path)
        load_model_manual(state_dict['state_dict'], model)
        print("Model loaded successfully")
        
    except Exception as e:
        print(f'ERROR: Model loading failed: {e}')
        return 1
    
    # Setup generator
    generator = config.get_generator(model, cfg, device=device)
    
    # Setup DPSR
    dpsr = DPSR(res=(cfg['generation']['psr_resolution'], 
                     cfg['generation']['psr_resolution'], 
                     cfg['generation']['psr_resolution']), 
                sig=cfg['generation']['psr_sigma']).to(device)
    
    # Statistics tracking
    stats = {
        'total_samples': 0,
        'successful_generations': 0,
        'failed_generations': 0,
        'processing_times': [],
        'config_used': args.config,
        'model_used': model_path,
        'timestamp': timestamp,
        'device': str(device)
    }
    
    sample_metadata = []
    
    print("\nStarting inference...")
    model.eval()
    
    with torch.no_grad():
        for it, data in enumerate(tqdm(test_loader, desc="Generating crowns")):
            
            # Limit number of samples if specified
            if args.num_samples and it >= args.num_samples:
                break
            
            start_time = time.time()
            
            try:
                # Get sample info
                idx = data['idx'].item()
                
                try:
                    model_dict = dataset.get_model_dict(idx)
                except AttributeError:
                    model_dict = {'model': str(idx), 'category': 'crown'}
                
                modelname = model_dict['model']
                category_id = model_dict.get('category', 'crown')
                
                # Generate mesh
                out = generator.generate_mesh(data)
                vertices, faces, points, normals, generation_stats = out
                
                # Save mesh
                mesh_filename = f"{modelname}_mesh.off"
                mesh_path = mesh_dir / mesh_filename
                export_mesh(str(mesh_path), scale2onet(vertices), faces)
                
                # Save point cloud
                pc_filename = f"{modelname}_pointcloud.ply"
                pc_path = pointcloud_dir / pc_filename
                export_pointcloud(str(pc_path), scale2onet(points), normals)
                
                # Save input point cloud
                if 'inputs' in data:
                    input_points = data['inputs'].to(device)
                    input_filename = f"{modelname}_input.ply"
                    input_path = input_dir / input_filename
                    export_pointcloud(str(input_path), scale2onet(input_points))
                
                processing_time = time.time() - start_time
                
                # Track metadata
                sample_info = {
                    'idx': idx,
                    'modelname': modelname,
                    'category': category_id,
                    'mesh_file': mesh_filename,
                    'pointcloud_file': pc_filename,
                    'input_file': f"{modelname}_input.ply" if 'inputs' in data else None,
                    'processing_time': processing_time,
                    'vertices_count': len(vertices),
                    'faces_count': len(faces),
                    'points_count': len(points)
                }
                sample_info.update(generation_stats)
                sample_metadata.append(sample_info)
                
                stats['successful_generations'] += 1
                stats['processing_times'].append(processing_time)
                
            except Exception as e:
                print(f"\nWARNING: Failed to process sample {it} (idx {idx}): {e}")
                stats['failed_generations'] += 1
                continue
            
            stats['total_samples'] += 1
    
    # Calculate final statistics
    if stats['processing_times']:
        stats['avg_processing_time'] = np.mean(stats['processing_times'])
        stats['total_processing_time'] = np.sum(stats['processing_times'])
    else:
        stats['avg_processing_time'] = 0
        stats['total_processing_time'] = 0
    
    # Save metadata
    with open(metadata_dir / 'inference_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save sample metadata as CSV and JSON
    if sample_metadata:
        df = pd.DataFrame(sample_metadata)
        df.to_csv(metadata_dir / 'samples_metadata.csv', index=False)
        df.to_json(metadata_dir / 'samples_metadata.json', orient='records', indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_root}")
    print(f"Total samples processed: {stats['total_samples']}")
    print(f"Successful generations: {stats['successful_generations']}")
    print(f"Failed generations: {stats['failed_generations']}")
    print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
    print(f"Total processing time: {stats['total_processing_time']:.2f}s")
    
    print(f"\nGenerated files:")
    print(f"  - Meshes: {len(list(mesh_dir.glob('*.off')))} files in {mesh_dir}")
    print(f"  - Point clouds: {len(list(pointcloud_dir.glob('*.ply')))} files in {pointcloud_dir}")
    print(f"  - Input clouds: {len(list(input_dir.glob('*.ply')))} files in {input_dir}")
    print(f"  - Metadata: {len(list(metadata_dir.glob('*')))} files in {metadata_dir}")
    
    # Create a README file
    readme_content = f"""# Crown Inference Results

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Config used: {args.config}
Model used: {model_path}
Dataset split: {args.split}
Device: {device}

## Directory Structure

- `meshes/`: Generated 3D meshes in .off format
- `pointclouds/`: Generated point clouds in .ply format
- `inputs/`: Input point clouds in .ply format
- `metadata/`: Processing statistics and sample information

## Statistics

- Total samples processed: {stats['total_samples']}
- Successful generations: {stats['successful_generations']}
- Failed generations: {stats['failed_generations']}
- Average processing time: {stats['avg_processing_time']:.2f}s per sample

## Files

### Meshes ({len(list(mesh_dir.glob('*.off')))} files)
Generated 3D meshes that can be viewed in any mesh viewer (MeshLab, Blender, etc.)

### Point Clouds ({len(list(pointcloud_dir.glob('*.ply')))} files)
Generated oriented point clouds with normal vectors

### Inputs ({len(list(input_dir.glob('*.ply')))} files)
Original input point clouds used for generation

### Metadata
- `inference_stats.json`: Overall processing statistics
- `samples_metadata.csv/json`: Per-sample information and statistics

## Usage

All generated files can be downloaded and viewed locally:
- .off files: Use MeshLab, Blender, or similar 3D viewers
- .ply files: Use MeshLab, CloudCompare, or Python (Open3D, PyVista)
"""
    
    with open(output_root / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"\nREADME.md created with detailed information")
    print(f"\nAll files ready for download from: {output_root}")
    
    return 0


if __name__ == '__main__':
    exit(main())