#!/usr/bin/env python3
"""
Inference Script for Yize Point Cloud Data

This script processes all .ply files in data/point_cloud_yize/ and generates
meshes using the trained crown generation model.

Usage:
    python infer_yize_data.py [--config CONFIG] [--model MODEL] [--output_dir OUTPUT_DIR]
"""

import torch
import numpy as np
import argparse
import time
import os
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from tqdm import tqdm

# Import project modules
from src.utils import load_config, load_model_manual, scale2onet, export_mesh, export_pointcloud
from src.model import Encode2Points
from src.dpsr import DPSR
from src import config

# Point cloud loading utilities
import open3d as o3d


def load_pointcloud(file_path, num_points=5000):
    """Load and preprocess a point cloud from PLY file."""
    try:
        # Load using Open3D
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            raise ValueError("Empty point cloud")
        
        # Normalize to unit sphere (center only, do not scale down)
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to match crown training data scale (approximately [-40, +40])
        scale = np.max(np.abs(points))
        if scale > 0:
            # Scale to approximately [-20, +20] range to match crown data
            points = points / scale * 20.0
        
        # Subsample or pad to target number of points
        if len(points) > num_points:
            # Random subsampling
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        elif len(points) < num_points:
            # Pad by repeating points
            num_repeats = num_points // len(points) + 1
            points = np.tile(points, (num_repeats, 1))[:num_points]
        
        return torch.from_numpy(points).float()
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def generate_mesh_from_points(generator, points, device):
    """Generate mesh from input points using the proper generator."""
    # Convert points to proper data format expected by generator
    data = {
        'inputs': points.unsqueeze(0).to(device)  # Add batch dimension
    }
    
    # Use the proper generator
    return generator.generate_mesh(data)


def main():
    parser = argparse.ArgumentParser(description='Yize Point Cloud Inference')
    parser.add_argument('--config', type=str, default='configs/crown_training.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--output_dir', type=str, default='yize_inference_results',
                        help='Output directory for generated data')
    parser.add_argument('--data_dir', type=str, default='data/point_cloud_yize',
                        help='Directory containing PLY files')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--num_points', type=int, default=5000,
                        help='Number of points to use from each point cloud')
    
    args = parser.parse_args()
    
    print("Yize Point Cloud Inference Script")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Points per cloud: {args.num_points}")
    
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
    output_root = Path(args.output_dir) / f"yize_inference_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to: {output_root}")
    
    # Setup output directories
    mesh_dir = output_root / "meshes"
    pointcloud_dir = output_root / "pointclouds"
    input_dir = output_root / "inputs"
    metadata_dir = output_root / "metadata"
    
    for dir_path in [mesh_dir, pointcloud_dir, input_dir, metadata_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Find all PLY files
    data_path = Path(args.data_dir)
    ply_files = sorted(list(data_path.glob("*.ply")))
    
    if not ply_files:
        print(f"ERROR: No PLY files found in {data_path}")
        return 1
    
    print(f"Found {len(ply_files)} PLY files to process")
    
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
        state_dict = torch.load(model_path, map_location=device)
        load_model_manual(state_dict['state_dict'], model)
        print("Model loaded successfully")
        
    except Exception as e:
        print(f'ERROR: Model loading failed: {e}')
        return 1
    
    # Setup generator using the proper config function
    generator = config.get_generator(model, cfg, device=device)
    
    # Statistics tracking
    stats = {
        'total_files': len(ply_files),
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
    
    for i, ply_file in enumerate(tqdm(ply_files, desc="Processing point clouds")):
        start_time = time.time()
        
        try:
            # Extract filename without extension for output naming
            filename = ply_file.stem
            
            # Load point cloud
            points = load_pointcloud(ply_file, args.num_points)
            if points is None:
                print(f"Failed to load {ply_file}")
                stats['failed_generations'] += 1
                continue
            
            # Save input point cloud copy
            input_path = input_dir / f"{filename}_input.ply"
            export_pointcloud(str(input_path), points.numpy())
            
            # Generate mesh
            vertices, faces, output_points, normals, generation_stats = generate_mesh_from_points(generator, points, device)
            
            if len(vertices) == 0 or len(faces) == 0:
                print(f"Empty mesh generated for {filename}")
                stats['failed_generations'] += 1
                continue
            
            # Save mesh
            mesh_filename = f"{filename}_mesh.off"
            mesh_path = mesh_dir / mesh_filename
            export_mesh(str(mesh_path), scale2onet(vertices), faces)
            
            # Save output point cloud
            pc_filename = f"{filename}_pointcloud.ply"
            pc_path = pointcloud_dir / pc_filename
            export_pointcloud(str(pc_path), scale2onet(output_points), normals)
            
            processing_time = time.time() - start_time
            
            # Track metadata
            sample_info = {
                'filename': filename,
                'input_file': ply_file.name,
                'mesh_file': mesh_filename,
                'pointcloud_file': pc_filename,
                'input_copy_file': f"{filename}_input.ply",
                'processing_time': processing_time
            }
            sample_info.update(generation_stats)
            sample_metadata.append(sample_info)
            
            stats['successful_generations'] += 1
            stats['processing_times'].append(processing_time)
            
        except Exception as e:
            print(f"\nWARNING: Failed to process {ply_file}: {e}")
            stats['failed_generations'] += 1
            continue
    
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
    
    # Save sample metadata
    if sample_metadata:
        df = pd.DataFrame(sample_metadata)
        df.to_csv(metadata_dir / 'samples_metadata.csv', index=False)
        df.to_json(metadata_dir / 'samples_metadata.json', orient='records', indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_root}")
    print(f"Total files found: {stats['total_files']}")
    print(f"Successful generations: {stats['successful_generations']}")
    print(f"Failed generations: {stats['failed_generations']}")
    print(f"Success rate: {100 * stats['successful_generations'] / stats['total_files']:.1f}%")
    print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
    print(f"Total processing time: {stats['total_processing_time']:.2f}s")
    
    print(f"\nGenerated files:")
    print(f"  - Meshes: {len(list(mesh_dir.glob('*.off')))} files in {mesh_dir}")
    print(f"  - Point clouds: {len(list(pointcloud_dir.glob('*.ply')))} files in {pointcloud_dir}")
    print(f"  - Input copies: {len(list(input_dir.glob('*.ply')))} files in {input_dir}")
    print(f"  - Metadata: {len(list(metadata_dir.glob('*')))} files in {metadata_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())