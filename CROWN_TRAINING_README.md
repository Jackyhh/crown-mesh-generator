# Crown Mesh Generation Training Pipeline

This document describes the complete pipeline for training a crown generation model using the Shape As Points framework with dental crown STL data.

## Overview

The pipeline converts dental crown STL files into the required PSR (Poisson Surface Reconstruction) format and trains a neural network to generate crown meshes from point clouds. This implementation is specifically designed for dental crown generation using differentiable Poisson surface reconstruction.

## Dataset Summary

**Processed Dataset Statistics:**
- **Total Samples:** 219 crowns
- **Training Set:** 175 samples (80%)  
- **Validation Set:** 21 samples (10%)
- **Test Set:** 23 samples (10%)
- **Point Cloud Size:** 100,000 points per mesh
- **PSR Grid Resolution:** 128³
- **Data Format:** Compatible with ShapeNet PSR format

## Quick Start

### 1. Data Preprocessing (Already Completed)

The crown STL data has been preprocessed and is ready for training:

```bash
# The preprocessing script has already been run:
# python preprocess_crowns.py
# 
# Generated dataset location: data/crown_psr/
# Dataset structure matches ShapeNet PSR format
```

### 2. Train the Crown Model

```bash
# Start training with the crown-specific configuration
python train_crowns.py

# Or use the base training script directly
python train.py configs/crown_training.yaml
```

### 3. Generate Crown Meshes

After training, generate new crown meshes:

```bash
# Generate meshes using trained model
python generate_crowns.py

# Or use the base generation script
python generate.py configs/crown_training.yaml --model_file out/crown/training/model_best.pt
```

## Detailed Pipeline

### Preprocessing Pipeline

The preprocessing converts STL files to the required format:

**Input:** `data/crown_stl_01-16_1001_1219/00000001/pre_jaw_crown*/Crown.stl`
**Output:** `data/crown_psr/00000001/crown*/[pointcloud.npz, psr.npz]`

**Process:**
1. Load STL mesh using trimesh
2. Normalize mesh to unit cube ([-0.9, 0.9])
3. Sample 100K surface points with normals using PyTorch3D
4. Compute PSR field using DPSR (128³ resolution)
5. Save in compressed NPZ format (float16 for efficiency)

### Training Configuration

**Model Architecture:**
- **Encoder:** LocalPoolPointnet with 3D UNet
- **Grid Resolution:** 64³ feature grid  
- **Hidden Dimensions:** 64 (increased from default 32)
- **UNet Levels:** 4 (one more than default)
- **Feature Maps:** 64
- **PSR Grid:** 128³

**Training Parameters:**
- **Batch Size:** 8 (reduced due to increased model capacity)
- **Learning Rate:** 3e-4
- **Input Points:** 5,000 per sample
- **Noise Level:** 0.01 (reduced for cleaner crowns)
- **Total Epochs:** 100,000
- **Validation Every:** 2,000 iterations

### Key Features for Crown Generation

1. **Higher Capacity Model:** Increased hidden dimensions and feature maps for detailed crown geometry
2. **Finer Grid Resolution:** 64³ feature grid for better spatial detail capture  
3. **Reduced Noise:** Lower noise level (0.01 vs 0.025) for cleaner dental surfaces
4. **More Input Points:** 5,000 vs 3,000 points for better shape representation
5. **Additional UNet Level:** 4 levels for multi-scale feature processing

## File Structure

```
shape_as_points/
├── data/
│   ├── crown_stl_01-16_1001_1219/        # Original STL files
│   └── crown_psr/                        # Processed PSR dataset
│       ├── 00000001/                     # Crown category
│       │   ├── crown00001001/
│       │   │   ├── pointcloud.npz        # Surface points + normals
│       │   │   └── psr.npz              # PSR field (128³)
│       │   └── ...
│       ├── metadata.yaml                 # Dataset metadata
│       └── 00000001/                     # Train/val/test splits
│           ├── train.lst
│           ├── val.lst
│           └── test.lst
├── configs/
│   └── crown_training.yaml              # Crown-specific training config
├── preprocess_crowns.py                 # STL to PSR preprocessing
├── train_crowns.py                      # Crown training wrapper
└── generate_crowns.py                   # Crown generation wrapper
```

## Usage Examples

### Training from Scratch

```bash
# Start fresh training
python train_crowns.py --config configs/crown_training.yaml --gpu 0

# Resume from checkpoint
python train_crowns.py --resume out/crown/training/model_latest.pt
```

### Generating Results

```bash
# Generate test set results
python generate_crowns.py --model out/crown/training/model_best.pt

# Generate with specific number of samples
python generate_crowns.py --num_samples 20
```

### Monitoring Training

Training outputs will be in `out/crown/training/`:
- `model_best.pt` - Best model checkpoint
- `model_latest.pt` - Latest model checkpoint  
- `logs/` - Training logs and metrics
- `generation/` - Generated meshes during training

## Expected Results

**Training Time:** ~24-48 hours on modern GPU (depends on hardware)
**Memory Usage:** ~8-12GB GPU memory (batch size 8)
**Model Size:** ~50-100MB

**Generated Outputs:**
- High-quality crown meshes in PLY format
- Point clouds with oriented normals
- PSR fields for further processing

## Troubleshooting

### Common Issues

1. **Out of Memory:** Reduce batch size in `configs/crown_training.yaml`
2. **Slow Training:** Check GPU utilization, consider reducing grid resolution
3. **Poor Quality:** Increase training epochs or adjust learning rate

### Validation

Check dataset integrity:
```bash
# Verify dataset structure
find data/crown_psr -name "*.npz" | wc -l  # Should show 438 (219*2)

# Check first processed sample
python -c "import numpy as np; data=np.load('data/crown_psr/00000001/crown00001001/pointcloud.npz'); print(f'Points: {data[\"points\"].shape}, Normals: {data[\"normals\"].shape}')"
```

## Advanced Configuration

### Adjusting Model Complexity

Edit `configs/crown_training.yaml`:
- Increase `c_dim` for more model capacity
- Adjust `grid_resolution` for spatial detail vs memory trade-off  
- Modify `f_maps` in unet3d_kwargs for feature richness

### Dataset Modifications

Re-run preprocessing with different parameters:
```bash
# Higher resolution PSR field
python preprocess_crowns.py --resolution 256 --num_points 200000

# Process subset for testing
python preprocess_crowns.py --max_files 50
```

## Citation

If you use this crown generation pipeline, please cite the original Shape As Points paper:

```bibtex
@inproceedings{peng2021shape,
  title={Shape As Points: A Differentiable Poisson Solver},
  author={Peng, Songyou and Niemeyer, Michael and Mescheder, Lars and Pollefeys, Marc and Geiger, Andreas},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```