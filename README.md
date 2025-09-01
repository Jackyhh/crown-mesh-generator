# Crown Forge: A Differentiable Poisson Solver for Crown-like 3D Surface Reconstruction

**Crown Forge** is a specialized 3D crown mesh generation system built upon the "Shape As Points: A Differentiable Poisson Solver" framework. This project enables high-quality reconstruction of crown-like 3D surfaces from unoriented point clouds using deep learning and differentiable Poisson surface reconstruction (DPSR).

## About

Crown Forge leverages a neural architecture that combines:

- **Deep Point Cloud Encoding**: Uses LocalPoolPointnet with 3D UNet for robust feature extraction from noisy, unoriented crown point clouds
- **Differentiable Poisson Surface Reconstruction**: Converts predicted oriented points to smooth, watertight crown meshes via FFT-based Poisson solving  
- **Multi-Resolution Processing**: Operates at 128³ grid resolution with 4-level UNet architecture for capturing fine crown geometric details
- **Robust Training**: Handles real-world crown data with noise tolerance (0.01 noise level) and 5000-point input processing

### Key Technical Features

- **Input**: Unoriented 3D point clouds (5000 points) representing crown-like structures
- **Output**: High-quality watertight triangle meshes suitable for manufacturing and analysis
- **Architecture**: Enhanced encoder-decoder with increased capacity (64-dim features, 4-level UNet)
- **Training Data**: PSR-supervised learning on crown-specific datasets
- **Inference**: Both conditional generation (from test data) and mesh reconstruction capabilities

The system is particularly optimized for crown-like geometries with smooth surfaces and intricate details, making it suitable for dental, architectural crown moldings, or similar applications requiring high-fidelity 3D reconstruction.

## Environment Setup

For environment setup, refer to the original ['shape_as_points'](https://github.com/autonomousvision/shape_as_points) project installation instructions.

## Quick Start - Crown Model Training

The main entry point for training crown generation models is the `start_training.sh` script:

```bash
bash start_training.sh
```

This script will:
- Set up the training environment 
- Initialize training with crown-specific configurations
- Handle model checkpointing and logging

## Training Configuration

Training configurations are managed through YAML files in the `configs/` directory. You can modify these files to:
- Adjust model parameters
- Set training hyperparameters  
- Configure data paths
- Customize output directories

## Data Preparation

Prepare your crown point cloud data in the appropriate format before training. The framework expects oriented point clouds for optimal reconstruction quality.

## Original Shape As Points Framework

This project is based on the "Shape As Points: A Differentiable Poisson Solver" paper (NeurIPS 2021 Oral):

**Citation:**
```bibtex
@inproceedings{Peng2021SAP,
 author    = {Peng, Songyou and Jiang, Chiyu "Max" and Liao, Yiyi and Niemeyer, Michael and Pollefeys, Marc and Geiger, Andreas},
 title     = {Shape As Points: A Differentiable Poisson Solver},
 booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
 year      = {2021}}
```

For detailed technical information about the underlying DPSR framework, refer to the [original project repository](https://github.com/autonomousvision/shape_as_points).