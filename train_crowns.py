#!/usr/bin/env python3
"""
Crown Training Script

A wrapper script to train the crown generation model with the preprocessed crown dataset.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import logging
from datetime import datetime

def setup_logging(log_file):
    """Setup logging to both console and file."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger

def main():
    parser = argparse.ArgumentParser(description='Train crown generation model')
    parser.add_argument('--config', type=str, 
                        default='configs/crown_training.yaml',
                        help='Training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path (default: logs/crown_training_TIMESTAMP.log)')
    
    args = parser.parse_args()
    
    # Setup log file path
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"logs/crown_training_{timestamp}.log"
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info(f"Logging to file: {args.log_file}")
    
    # Set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Check if dataset is ready
    crown_psr_path = Path('data/crown_psr')
    if not crown_psr_path.exists():
        logger.error("Crown PSR dataset not found!")
        logger.error("Please run the preprocessing script first:")
        logger.error("  python preprocess_crowns.py")
        return 1
    
    # Check if metadata exists
    metadata_file = crown_psr_path / 'metadata.yaml'
    if not metadata_file.exists():
        logger.error("Dataset metadata not found!")
        logger.error("Please run the preprocessing script to completion.")
        return 1
    
    # Check if training config exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    logger.info("Crown Model Training")
    logger.info("=" * 50)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Dataset: {crown_psr_path}")
    logger.info(f"GPU: {args.gpu}")
    
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
    
    # Build command
    cmd = [
        '/root/anaconda3/envs/py310-dmc/bin/python',
        'train.py',
        str(config_path)
    ]
    
    if args.resume:
        cmd.extend(['--resume', args.resume])
    
    logger.info(f"\nExecuting: {' '.join(cmd)}")
    logger.info("")
    
    # Run training with output capture
    try:
        # Use Popen to capture output in real-time
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output to both console and log file
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output = output.rstrip()
                logger.info(output)
        
        # Wait for process to complete
        return_code = process.poll()
        
        if return_code == 0:
            logger.info("\nTraining completed successfully!")
            return 0
        else:
            logger.error(f"\nTraining failed with exit code: {return_code}")
            return return_code
            
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        if 'process' in locals():
            process.terminate()
        return 1
    except Exception as e:
        logger.error(f"\nUnexpected error during training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())