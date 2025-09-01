#!/usr/bin/env python3

print("Debug script starting...")

import torch
print("torch imported")

import trimesh
print("trimesh imported")

import shutil, argparse, time, os, glob
print("basic modules imported")

import numpy as np; np.set_printoptions(precision=4)
print("numpy imported")

import open3d as o3d
print("open3d imported")

from torch.utils.tensorboard import SummaryWriter
print("tensorboard imported")

from torchvision.utils import save_image
from torchvision.io import write_video
print("torchvision imported")

from src.optimization import Trainer
print("src.optimization imported")

from src.utils import load_config, update_config, initialize_logger, \
    get_learning_rate_schedules, adjust_learning_rate, AverageMeter,\
         update_optimizer, export_pointcloud
print("src.utils imported")

from skimage import measure
print("skimage imported")

from plyfile import PlyData
print("plyfile imported")

from pytorch3d.ops import sample_points_from_meshes
print("pytorch3d.ops imported")

from pytorch3d.io import load_objs_as_meshes
print("pytorch3d.io imported")

from pytorch3d.structures import Meshes
print("pytorch3d.structures imported")

print("All imports successful in debug script")

def main():
    print("Starting main function...")
    parser = argparse.ArgumentParser(description='MNIST toy experiment')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')    
    parser.add_argument('--seed', type=int, default=1457, metavar='S', 
                        help='random seed')
    
    print("Parsing arguments...")
    args, unknown = parser.parse_known_args() 
    print("Loading config...")
    cfg = load_config(args.config, 'configs/default.yaml')
    print("Config loaded successfully")
    
    print("Script completed without crash!")

if __name__ == '__main__':
    print("Script entry point...")
    main()