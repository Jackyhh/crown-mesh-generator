# Crown Mesh Generator

This project uses the "Shape As Points: A Differentiable Poisson Solver" framework for training crown mesh generation models from point cloud data.

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