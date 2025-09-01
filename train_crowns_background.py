#!/usr/bin/env python3
"""
Crown Training Script - Background Version

A wrapper script to train the crown generation model in the background using nohup.
This version continues running even if the terminal connection is lost.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import signal

def main():
    parser = argparse.ArgumentParser(description='Train crown generation model in background')
    parser.add_argument('--config', type=str, 
                        default='configs/crown_training.yaml',
                        help='Training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path (default: logs/crown_training_TIMESTAMP.log)')
    parser.add_argument('--pid-file', type=str, default=None,
                        help='PID file path (default: logs/crown_training_TIMESTAMP.pid)')
    
    args = parser.parse_args()
    
    # Setup log file path
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"logs/crown_training_{timestamp}.log"
    
    # Setup PID file path
    if args.pid_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.pid_file = f"logs/crown_training_{timestamp}.pid"
    
    # Create logs directory
    log_dir = Path(args.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset is ready
    crown_psr_path = Path('data/crown_psr')
    if not crown_psr_path.exists():
        print("ERROR: Crown PSR dataset not found!")
        print("Please run the preprocessing script first:")
        print("  python preprocess_crowns.py")
        return 1
    
    # Check if metadata exists
    metadata_file = crown_psr_path / 'metadata.yaml'
    if not metadata_file.exists():
        print("ERROR: Dataset metadata not found!")
        print("Please run the preprocessing script to completion.")
        return 1
    
    # Check if training config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        return 1
    
    print("Crown Model Training - Background Mode")
    print("=" * 50)
    print(f"Configuration: {args.config}")
    print(f"Dataset: {crown_psr_path}")
    print(f"GPU: {args.gpu}")
    print(f"Log file: {args.log_file}")
    print(f"PID file: {args.pid_file}")
    
    if args.resume:
        print(f"Resuming from: {args.resume}")
    
    # Build the background command
    cmd_parts = [
        'nohup',
        '/root/anaconda3/envs/py310-dmc/bin/python',
        'train_crowns.py',
        '--config', str(config_path),
        '--gpu', args.gpu,
        '--log-file', args.log_file
    ]
    
    if args.resume:
        cmd_parts.extend(['--resume', args.resume])
    
    # Redirect output and run in background
    cmd = ' '.join(cmd_parts) + f' > {args.log_file} 2>&1 &'
    
    print(f"\nExecuting background command:")
    print(cmd)
    print(f"\nStarting training in background...")
    
    # Execute the command
    try:
        # Use shell=True to handle the background redirection properly
        process = subprocess.Popen(cmd, shell=True)
        
        # Give it a moment to start
        import time
        time.sleep(2)
        
        # Get the actual training process PID by finding the python train_crowns.py process
        find_pid_cmd = "ps aux | grep 'python.*train_crowns.py' | grep -v grep | awk '{print $2}' | head -1"
        pid_result = subprocess.run(find_pid_cmd, shell=True, capture_output=True, text=True)
        
        if pid_result.stdout.strip():
            training_pid = pid_result.stdout.strip()
            
            # Save PID to file
            with open(args.pid_file, 'w') as f:
                f.write(training_pid)
            
            print(f"Training started successfully!")
            print(f"Process ID: {training_pid}")
            print(f"PID saved to: {args.pid_file}")
            print(f"Logs being written to: {args.log_file}")
            print(f"\nTo monitor progress:")
            print(f"  tail -f {args.log_file}")
            print(f"\nTo check if training is running:")
            print(f"  ps -p {training_pid}")
            print(f"\nTo stop training:")
            print(f"  kill {training_pid}")
            
            return 0
        else:
            print("ERROR: Could not determine training process ID")
            return 1
            
    except Exception as e:
        print(f"ERROR: Failed to start background training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())