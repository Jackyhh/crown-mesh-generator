import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import torch
import torch.optim as optim
import open3d as o3d

import numpy as np; np.set_printoptions(precision=4)
import shutil, argparse, time
from torch.utils.tensorboard import SummaryWriter

from src import config
from src.data import collate_remove_none, collate_stack_together, worker_init_fn
from src.training import Trainer
from src.model import Encode2Points
from src.utils import load_config, initialize_logger, \
AverageMeter, load_model_manual

def main():
    print("DEBUG: Starting main function")
    
    # Hardcode the config path for debugging
    config_path = 'configs/learning_based/noise_large/ours.yaml'
    print(f"DEBUG: Loading config from {config_path}")
    
    cfg = load_config(config_path, 'configs/default.yaml')
    print("DEBUG: Config loaded successfully")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"DEBUG: Using device: {device}")
    
    input_type = cfg['data']['input_type']
    batch_size = cfg['train']['batch_size']
    model_selection_metric = cfg['train']['model_selection_metric']
    print(f"DEBUG: batch_size={batch_size}, model_selection_metric={model_selection_metric}")

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)
    print("DEBUG: PyTorch version check passed")

    # boiler-plate
    if cfg['train']['timestamp']:
        cfg['train']['out_dir'] += '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    print(f"DEBUG: Output directory: {cfg['train']['out_dir']}")
    
    logger = initialize_logger(cfg)
    print("DEBUG: Logger initialized")
    
    torch.manual_seed(1)
    np.random.seed(1)
    print("DEBUG: Random seeds set")

    shutil.copyfile(config_path, os.path.join(cfg['train']['out_dir'], 'config.yaml'))
    print("DEBUG: Config file copied")

    logger.info("using GPU: " + torch.cuda.get_device_name(0))

    # TensorboardX writer
    tblogdir = os.path.join(cfg['train']['out_dir'], "tensorboard_log")
    if not os.path.exists(tblogdir):
        os.makedirs(tblogdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tblogdir)
    print("DEBUG: TensorBoard writer initialized")
    
    inputs = None
    print("DEBUG: Getting datasets...")
    train_dataset = config.get_dataset('train', cfg)
    print(f"DEBUG: Train dataset loaded with {len(train_dataset)} samples")
    
    val_dataset = config.get_dataset('val', cfg)
    print(f"DEBUG: Val dataset loaded with {len(val_dataset)} samples")
    
    vis_dataset = config.get_dataset('vis', cfg)
    print(f"DEBUG: Vis dataset loaded with {len(vis_dataset)} samples")
    
    collate_fn = collate_remove_none

    print("DEBUG: Creating data loaders...")
    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['train']['n_workers'], shuffle=True,
    collate_fn=collate_fn,
    worker_init_fn=worker_init_fn)
    print("DEBUG: Train loader created")

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=cfg['train']['n_workers_val'], shuffle=False,
    collate_fn=collate_remove_none,
    worker_init_fn=worker_init_fn)
    print("DEBUG: Val loader created")

    vis_loader = torch.utils.data.DataLoader(
        vis_dataset, batch_size=1, num_workers=cfg['train']['n_workers_val'], shuffle=False,
    collate_fn=collate_fn,
    worker_init_fn=worker_init_fn)
    print("DEBUG: Vis loader created")
    
    print("DEBUG: Creating model...")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(Encode2Points(cfg)).to(device)
    else:
        model = Encode2Points(cfg).to(device)
    print("DEBUG: Model created and moved to device")

    n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of parameters: %d'% n_parameter)
    print(f"DEBUG: Model has {n_parameter} parameters")
    
    # load model
    try:
        print("DEBUG: Trying to load existing model...")
        # load model
        state_dict = torch.load(os.path.join(cfg['train']['out_dir'], 'model.pt'))
        load_model_manual(state_dict['state_dict'], model)
            
        out = "Load model from iteration %d" % state_dict.get('it', 0)
        logger.info(out)
        print(f"DEBUG: {out}")
        # load point cloud
    except:
        print("DEBUG: No existing model found, starting fresh")
        state_dict = dict()

    print("DEBUG: Setting up optimizer...")
    LR = float(cfg['train']['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"DEBUG: Optimizer created with LR={LR}")
    
    print("DEBUG: About to create Trainer...")
    trainer = Trainer(cfg, optimizer, device=device)
    print("DEBUG: Trainer created successfully!")
    
    print("DEBUG: Testing first training step...")
    inputs = None
    batch = next(iter(train_loader))
    print("DEBUG: Got first batch")
    
    print("DEBUG: Calling train_step...")
    loss, loss_each = trainer.train_step(inputs, batch, model)
    print(f"DEBUG: First train_step completed! Loss: {loss}")
    
    print("DEBUG: Training initialization completed successfully!")

if __name__ == '__main__':
    main()