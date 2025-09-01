#!/usr/bin/env python3
"""
Crown STL to PSR Format Preprocessing Script

This script converts dental crown STL files to the format required by the Shape As Points training pipeline.
It generates pointcloud.npz and psr.npz files compatible with the ShapeNet PSR dataset format.

Input:  Crown STL files organized as: data/crown_stl_01-16_1001_1219/00000001/pre_jaw_crown*/Crown.stl
Output: PSR format files organized as: data/crown_psr/00000001/crown_XXXXX/{pointcloud.npz, psr.npz}
"""

import os
import numpy as np
import torch
from pathlib import Path
import argparse
import glob
from tqdm import tqdm
import trimesh
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
import tempfile

# Add src to path for DPSR imports
import sys
sys.path.append('/workspace/github/shape_as_points/src')
from dpsr import DPSR


def load_stl_mesh(stl_path, device='cuda'):
    """Load STL file and convert to PyTorch3D mesh format."""
    # Load with trimesh first to handle STL format
    mesh_trimesh = trimesh.load(stl_path)
    
    # Convert to PyTorch3D format
    vertices = torch.tensor(mesh_trimesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh_trimesh.faces, dtype=torch.long, device=device)
    
    # Create PyTorch3D mesh
    mesh = Meshes(verts=[vertices], faces=[faces])
    
    return mesh


def normalize_mesh(mesh):
    """Normalize mesh to unit cube centered at origin."""
    vertices = mesh.verts_packed()
    
    # Center at origin
    center = vertices.mean(0)
    vertices_centered = vertices - center
    
    # Scale to fit in unit cube with some margin
    scale = torch.max(torch.abs(vertices_centered))
    vertices_normalized = vertices_centered / scale * 0.9
    
    # Create new mesh with normalized vertices
    normalized_mesh = Meshes(verts=[vertices_normalized], faces=[mesh.faces_packed()])
    
    return normalized_mesh, center.cpu().numpy(), scale.cpu().numpy()


def sample_surface_points(mesh, num_points=100000):
    """Sample points and normals from mesh surface."""
    points, normals = sample_points_from_meshes(
        mesh, 
        num_samples=num_points, 
        return_normals=True
    )
    
    return points.squeeze(0), normals.squeeze(0)


def compute_psr_field(points, normals, resolution=128, sigma=2):
    """Compute PSR field using DPSR."""
    device = points.device
    
    # Initialize DPSR
    dpsr = DPSR(res=(resolution, resolution, resolution), sig=sigma).to(device)
    
    # Convert points to [0, 1] range as expected by DPSR
    points_normalized = (points + 1) / 2
    
    # Add batch dimension
    points_batch = points_normalized.unsqueeze(0)
    normals_batch = normals.unsqueeze(0)
    
    # Compute PSR field
    psr_field = dpsr(points_batch, normals_batch)
    
    return psr_field.squeeze(0)


def save_npz_files(points, normals, psr_field, output_dir):
    """Save pointcloud.npz and psr.npz files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save pointcloud data
    pointcloud_path = os.path.join(output_dir, 'pointcloud.npz')
    np.savez_compressed(
        pointcloud_path,
        points=points.cpu().numpy().astype(np.float16),
        normals=normals.cpu().numpy().astype(np.float16)
    )
    
    # Save PSR field
    psr_path = os.path.join(output_dir, 'psr.npz')
    np.savez_compressed(
        psr_path,
        psr=psr_field.cpu().numpy().astype(np.float16)
    )
    
    return pointcloud_path, psr_path


def process_crown_file(stl_path, output_dir, num_points=100000, resolution=128, device='cuda'):
    """Process a single crown STL file."""
    try:
        # Load and normalize mesh
        mesh = load_stl_mesh(stl_path, device)
        mesh_norm, center, scale = normalize_mesh(mesh)
        
        # Sample surface points and normals
        points, normals = sample_surface_points(mesh_norm, num_points)
        
        # Compute PSR field
        psr_field = compute_psr_field(points, normals, resolution)
        
        # Save files
        pc_path, psr_path = save_npz_files(points, normals, psr_field, output_dir)
        
        # Save metadata
        metadata = {
            'original_center': center,
            'original_scale': scale,
            'num_points': num_points,
            'psr_resolution': resolution
        }
        
        return True, metadata
        
    except Exception as e:
        print(f"Error processing {stl_path}: {str(e)}")
        return False, None


def create_dataset_splits(crown_dirs, output_base_dir, train_ratio=0.8, val_ratio=0.1):
    """Create train/val/test split files."""
    crown_ids = list(crown_dirs.keys())
    np.random.shuffle(crown_ids)
    
    n_total = len(crown_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = crown_ids[:n_train]
    val_ids = crown_ids[n_train:n_train + n_val]
    test_ids = crown_ids[n_train + n_val:]
    
    # Create category directory (using 00000001 as single category for crowns)
    category_dir = os.path.join(output_base_dir, '00000001')
    os.makedirs(category_dir, exist_ok=True)
    
    # Write split files
    splits = {
        'train.lst': train_ids,
        'val.lst': val_ids,
        'test.lst': test_ids
    }
    
    for split_name, ids in splits.items():
        split_path = os.path.join(category_dir, split_name)
        with open(split_path, 'w') as f:
            for crown_id in ids:
                f.write(f"{crown_id}\n")
    
    print(f"Dataset splits created:")
    print(f"  Train: {len(train_ids)} samples")
    print(f"  Val: {len(val_ids)} samples")  
    print(f"  Test: {len(test_ids)} samples")


def main():
    parser = argparse.ArgumentParser(description='Preprocess crown STL files to PSR format')
    parser.add_argument('--input_dir', type=str, 
                        default='/workspace/github/shape_as_points/data/crown_stl_01-16_1001_1219',
                        help='Input directory containing crown STL files')
    parser.add_argument('--output_dir', type=str,
                        default='/workspace/github/shape_as_points/data/crown_psr',
                        help='Output directory for processed PSR files')
    parser.add_argument('--num_points', type=int, default=100000,
                        help='Number of points to sample from each mesh')
    parser.add_argument('--resolution', type=int, default=128,
                        help='PSR grid resolution')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    print("Crown STL to PSR Format Preprocessing")
    print("="*50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Points per mesh: {args.num_points}")
    print(f"PSR resolution: {args.resolution}")
    print(f"Device: {args.device}")
    print()
    
    # Find all STL files
    stl_pattern = os.path.join(args.input_dir, "**", "*.stl")
    stl_files = glob.glob(stl_pattern, recursive=True)
    
    if args.max_files:
        stl_files = stl_files[:args.max_files]
    
    print(f"Found {len(stl_files)} STL files to process")
    
    if len(stl_files) == 0:
        print("No STL files found! Check the input directory.")
        return
    
    # Organize by crown ID
    crown_dirs = {}
    for stl_path in stl_files:
        # Extract crown ID from path: .../pre_jaw_crown00001037/Crown.stl -> crown00001037
        path_parts = Path(stl_path).parts
        crown_folder = next(part for part in path_parts if part.startswith('pre_jaw_crown'))
        crown_id = crown_folder.replace('pre_jaw_', '')  # Remove 'pre_jaw_' prefix
        crown_dirs[crown_id] = stl_path
    
    print(f"Processing {len(crown_dirs)} unique crowns")
    
    # Create output directory structure matching ShapeNet PSR
    os.makedirs(args.output_dir, exist_ok=True)
    category_dir = os.path.join(args.output_dir, '00000001')  # Single category for crowns
    os.makedirs(category_dir, exist_ok=True)
    
    # Process each crown
    successful = 0
    failed = 0
    
    with tqdm(crown_dirs.items(), desc="Processing crowns") as pbar:
        for crown_id, stl_path in pbar:
            pbar.set_description(f"Processing {crown_id}")
            
            # Create output directory for this crown
            crown_output_dir = os.path.join(category_dir, crown_id)
            
            # Skip if already processed
            if (os.path.exists(os.path.join(crown_output_dir, 'pointcloud.npz')) and
                os.path.exists(os.path.join(crown_output_dir, 'psr.npz'))):
                print(f"Skipping {crown_id} (already processed)")
                successful += 1
                continue
            
            # Process the crown
            success, metadata = process_crown_file(
                stl_path, crown_output_dir, 
                args.num_points, args.resolution, args.device
            )
            
            if success:
                successful += 1
                pbar.set_postfix(success=successful, failed=failed)
            else:
                failed += 1
                pbar.set_postfix(success=successful, failed=failed)
    
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Create dataset splits
    if successful > 0:
        print("\nCreating dataset splits...")
        create_dataset_splits(crown_dirs, args.output_dir)
        
        # Create metadata file
        metadata_path = os.path.join(args.output_dir, 'metadata.yaml')
        with open(metadata_path, 'w') as f:
            f.write(f"# Crown Dataset Metadata\n")
            f.write(f"dataset_name: crown_psr\n")
            f.write(f"num_categories: 1\n") 
            f.write(f"num_samples: {successful}\n")
            f.write(f"point_cloud_size: {args.num_points}\n")
            f.write(f"psr_resolution: {args.resolution}\n")
            f.write(f"categories:\n")
            f.write(f"  '00000001': dental_crown\n")
        
        print(f"Metadata saved to {metadata_path}")
    
    print(f"\nDataset ready for training!")
    print(f"Use this path in your config: {args.output_dir}")


if __name__ == "__main__":
    main()