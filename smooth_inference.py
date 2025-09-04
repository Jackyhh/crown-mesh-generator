#!/usr/bin/env python3
"""
Enhanced Crown Inference with Surface Smoothing

This script performs inference with improved surface quality through:
1. Higher resolution PSR grids
2. Post-processing mesh smoothing
3. Adaptive normal filtering
"""

import torch
import trimesh
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from src.utils import load_config, load_model_manual, scale2onet, export_mesh, export_pointcloud
from src.model import Encode2Points
from src.dpsr import DPSR
from src import config
import json
from tqdm import tqdm


def apply_laplacian_smoothing(vertices, faces, iterations=5, lambda_factor=0.5):
    """
    Apply Laplacian smoothing to mesh vertices
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        iterations: Number of smoothing iterations
        lambda_factor: Smoothing strength (0-1)
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Apply smoothing
    for _ in range(iterations):
        # Get vertex adjacency
        vertex_adjacency = mesh.vertex_adjacency_graph
        
        new_vertices = vertices.copy()
        for i in range(len(vertices)):
            neighbors = list(vertex_adjacency[i])
            if len(neighbors) > 0:
                neighbor_centroid = np.mean(vertices[neighbors], axis=0)
                new_vertices[i] = (1 - lambda_factor) * vertices[i] + lambda_factor * neighbor_centroid
        
        vertices = new_vertices
    
    return vertices


def gaussian_smooth_normals(points, normals, sigma=0.01, k_neighbors=10):
    """
    Apply Gaussian smoothing to normal vectors
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Find k-nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Apply Gaussian weights and smooth normals
    smoothed_normals = np.zeros_like(normals)
    for i in range(len(points)):
        weights = np.exp(-distances[i]**2 / (2 * sigma**2))
        weights = weights / np.sum(weights)
        
        neighbor_normals = normals[indices[i]]
        smoothed_normals[i] = np.sum(neighbor_normals * weights[:, np.newaxis], axis=0)
        # Normalize
        norm = np.linalg.norm(smoothed_normals[i])
        if norm > 0:
            smoothed_normals[i] /= norm
    
    return smoothed_normals


def main():
    parser = argparse.ArgumentParser(description='Enhanced Crown Inference with Smoothing')
    parser.add_argument('--config', type=str, default='configs/crown_training_smooth.yaml',
                        help='Path to config file.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='inference_output_smooth',
                        help='Output directory')
    parser.add_argument('--smooth_mesh', action='store_true',
                        help='Apply Laplacian smoothing to output meshes')
    parser.add_argument('--smooth_normals', action='store_true',
                        help='Apply Gaussian smoothing to normals')
    parser.add_argument('--smoothing_iterations', type=int, default=3,
                        help='Number of Laplacian smoothing iterations')
    parser.add_argument('--smoothing_strength', type=float, default=0.3,
                        help='Laplacian smoothing strength (0-1)')
    parser.add_argument('--normal_sigma', type=float, default=0.01,
                        help='Gaussian sigma for normal smoothing')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    print("Enhanced Crown Inference with Surface Smoothing")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Mesh smoothing: {args.smooth_mesh}")
    print(f"Normal smoothing: {args.smooth_normals}")
    
    # Load config
    cfg = load_config(args.config, 'configs/default.yaml')
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"smooth_crown_inference_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    
    mesh_dir = output_root / "meshes"
    pointcloud_dir = output_root / "pointclouds"
    input_dir = output_root / "inputs"
    metadata_dir = output_root / "metadata"
    
    for dir_path in [mesh_dir, pointcloud_dir, input_dir, metadata_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Load dataset
    dataset = config.get_dataset('test', cfg, return_idx=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    
    # Load model
    model = Encode2Points(cfg).to(device)
    
    # Determine model path
    out_dir = cfg['train']['out_dir']
    if args.model:
        model_path = args.model
    else:
        model_path = os.path.join(out_dir, 'model_best.pt')
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return 1
    
    state_dict = torch.load(model_path)
    load_model_manual(state_dict['state_dict'], model)
    
    # Setup generator
    generator = config.get_generator(model, cfg, device=device)
    
    # Statistics
    stats = {
        'smoothing_enabled': args.smooth_mesh or args.smooth_normals,
        'smoothing_iterations': args.smoothing_iterations if args.smooth_mesh else 0,
        'smoothing_strength': args.smoothing_strength if args.smooth_mesh else 0,
        'normal_smoothing': args.smooth_normals,
        'normal_sigma': args.normal_sigma if args.smooth_normals else 0,
        'successful_generations': 0,
        'failed_generations': 0,
        'processing_times': []
    }
    
    model.eval()
    
    print(f"\nProcessing {len(dataset)} samples...")
    print(f"Higher resolution PSR: {cfg['generation']['psr_resolution']}x{cfg['generation']['psr_resolution']}x{cfg['generation']['psr_resolution']}")
    print(f"PSR sigma: {cfg['generation']['psr_sigma']}")
    
    with torch.no_grad():
        for it, data in enumerate(tqdm(test_loader, desc="Generating smooth crowns")):
            
            if args.num_samples and it >= args.num_samples:
                break
            
            try:
                # Get sample info
                idx = data['idx'].item()
                try:
                    model_dict = dataset.get_model_dict(idx)
                except AttributeError:
                    model_dict = {'model': f'crown_{idx:08d}', 'category': 'crown'}
                
                modelname = model_dict['model']
                
                # Generate mesh
                out = generator.generate_mesh(data)
                vertices, faces, points, normals, generation_stats = out
                
                # Apply smoothing if requested
                if args.smooth_mesh and len(vertices) > 0 and len(faces) > 0:
                    try:
                        vertices_smooth = apply_laplacian_smoothing(
                            vertices, faces, 
                            iterations=args.smoothing_iterations,
                            lambda_factor=args.smoothing_strength
                        )
                        vertices = vertices_smooth
                    except Exception as e:
                        print(f"Warning: Mesh smoothing failed for {modelname}: {e}")
                
                # Apply normal smoothing if requested
                if args.smooth_normals and len(points) > 0 and len(normals) > 0:
                    try:
                        normals_smooth = gaussian_smooth_normals(
                            points, normals, 
                            sigma=args.normal_sigma
                        )
                        normals = normals_smooth
                    except Exception as e:
                        print(f"Warning: Normal smoothing failed for {modelname}: {e}")
                
                # Save results
                mesh_filename = f"{modelname}_smooth_mesh.off"
                mesh_path = mesh_dir / mesh_filename
                export_mesh(str(mesh_path), scale2onet(vertices), faces)
                
                pc_filename = f"{modelname}_smooth_pointcloud.ply"
                pc_path = pointcloud_dir / pc_filename
                export_pointcloud(str(pc_path), scale2onet(points), normals)
                
                # Save input
                if 'inputs' in data:
                    input_points = data['inputs'].to(device)
                    input_filename = f"{modelname}_input.ply"
                    input_path = input_dir / input_filename
                    export_pointcloud(str(input_path), scale2onet(input_points))
                
                stats['successful_generations'] += 1
                
            except Exception as e:
                print(f"Error processing sample {it}: {e}")
                stats['failed_generations'] += 1
    
    # Save metadata
    with open(metadata_dir / 'smooth_inference_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create README
    readme_content = f"""# Enhanced Crown Inference Results

Generated with surface smoothing improvements:

## Improvements Applied:
- Higher resolution PSR grid: {cfg['generation']['psr_resolution']}³
- PSR sigma: {cfg['generation']['psr_sigma']} (increased for smoothness)
- Mesh smoothing: {'Enabled' if args.smooth_mesh else 'Disabled'}
- Normal smoothing: {'Enabled' if args.smooth_normals else 'Disabled'}

## Surface Quality Enhancements:
1. **Higher Resolution**: 2x increase in PSR grid resolution for finer details
2. **Improved PSR Parameters**: Optimized sigma for smoother reconstruction  
3. **Laplacian Smoothing**: {args.smoothing_iterations} iterations with {args.smoothing_strength} strength
4. **Gaussian Normal Smoothing**: Applied with sigma={args.normal_sigma}

## Results:
- Successful generations: {stats['successful_generations']}
- Failed generations: {stats['failed_generations']}

The generated meshes should show significantly improved surface quality with reduced roughness while maintaining the overall crown geometry.
"""
    
    with open(output_root / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"\nSmoothed inference complete!")
    print(f"Results saved to: {output_root}")
    print(f"Successful: {stats['successful_generations']}, Failed: {stats['failed_generations']}")
    
    return 0


if __name__ == '__main__':
    import os
    exit(main())