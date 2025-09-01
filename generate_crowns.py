#!/usr/bin/env python3
"""
Crown Generation Script

Generate crown meshes from trained models or from input point clouds.
"""

import sys
import os
import subprocess
import argparse
import numpy as np
from pathlib import Path
import torch
from pytorch3d.io import save_ply
from pytorch3d.structures import Pointclouds


def main():
    parser = argparse.ArgumentParser(description='Generate crown meshes')
    parser.add_argument('--config', type=str, 
                        default='configs/crown_training.yaml',
                        help='Configuration file')
    parser.add_argument('--model', type=str, 
                        default='out/crown/training/model_best.pt',
                        help='Trained model checkpoint')
    parser.add_argument('--input', type=str, default=None,
                        help='Input point cloud file (.ply or .npz)')
    parser.add_argument('--output_dir', type=str, 
                        default='out/crown/generated',
                        help='Output directory for generated meshes')
    parser.add_argument('--dataset_split', type=str, 
                        choices=['train', 'val', 'test'], default='test',
                        help='Dataset split to generate from')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate from dataset')
    
    args = parser.parse_args()
    
    # Set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found: {model_path}")
        print("Please train a model first using:")
        print("  python train_crowns.py")
        return 1
    
    # Check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Crown Mesh Generation")
    print("=" * 50)
    print(f"Configuration: {args.config}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"GPU: {args.gpu}")
    
    if args.input:
        print(f"Input: {args.input}")
        # TODO: Implement single file inference
        print("Single file inference not yet implemented.")
        print("Use dataset generation instead.")
        return 1
    else:
        print(f"Dataset split: {args.dataset_split}")
        print(f"Number of samples: {args.num_samples}")
    
    # Build command for dataset generation
    cmd = [
        '/root/anaconda3/envs/py310-dmc/bin/python',
        'generate.py',
        str(config_path),
        '--model_file', str(model_path)
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    print()
    
    # Run generation
    try:
        result = subprocess.run(cmd, check=True, cwd=Path.cwd())
        print("\nGeneration completed successfully!")
        
        # List generated files
        generation_dir = Path('out/crown/training/generation')
        if generation_dir.exists():
            ply_files = list(generation_dir.glob('*.ply'))
            print(f"\nGenerated {len(ply_files)} mesh files in {generation_dir}")
            
            # Show first few files
            for i, ply_file in enumerate(ply_files[:5]):
                print(f"  {ply_file}")
            if len(ply_files) > 5:
                print(f"  ... and {len(ply_files) - 5} more files")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\nGeneration failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        return 1


def generate_from_pointcloud(input_file, model_path, config_path, output_path):
    """Generate mesh from a single point cloud file."""
    # This is a placeholder for single-file inference
    # Would need to implement the full inference pipeline
    print("Single file inference not implemented yet")
    return False


if __name__ == "__main__":
    sys.exit(main())