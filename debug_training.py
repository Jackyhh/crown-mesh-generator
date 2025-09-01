#!/usr/bin/env python3

import os
import sys
import torch
from src.utils import load_config
from src import config

def test_training_setup():
    print("=== Debug Training Setup ===")
    
    # Load config
    config_path = 'configs/learning_based/noise_large/ours.yaml'
    print(f"Loading config: {config_path}")
    
    cfg = load_config(config_path, 'configs/default.yaml')
    print("Config loaded successfully")
    print("Dataset path:", cfg['data'].get('path', 'NOT FOUND'))
    print("Dataset type:", cfg['data'].get('dataset', 'NOT FOUND'))
    
    # Test dataset loading
    print("Testing dataset loading...")
    try:
        print("Getting dataset...")
        train_dataset = config.get_dataset('train', cfg)
        print(f"Train dataset loaded: {len(train_dataset)} samples")
        
        # Test getting one sample
        print("Testing sample loading...")
        sample = train_dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print("Sample loaded successfully")
        
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test model creation  
    print("Testing model creation...")
    try:
        from src.model import Encode2Points
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        print("Creating model...")
        model = Encode2Points(cfg)
        print("Model created successfully")
        
        print("Moving model to device...")
        model = model.to(device)
        print("Model moved to device successfully")
        
        n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of parameters: {n_parameter}')
        
    except Exception as e:
        print(f"Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test data loader
    print("Testing data loader...")
    try:
        from src.data import collate_remove_none, worker_init_fn
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, num_workers=0, shuffle=True,
            collate_fn=collate_remove_none,
            worker_init_fn=worker_init_fn)
        
        print("Data loader created successfully")
        
        print("Testing first batch...")
        batch = next(iter(train_loader))
        print(f"Batch keys: {list(batch.keys())}")
        print("First batch loaded successfully")
        
    except Exception as e:
        print(f"Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=== All tests passed ===")
    return True

if __name__ == "__main__":
    test_training_setup()