import os
from pathlib import Path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import time
import yaml
import argparse
import pandas as pd
from src.data.dataset import (
    MILDataset,
)
from torch.utils.data import DataLoader
import torch
import lightning as L
from src.training.trainer import MammothTrainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, DeviceStatsMonitor
from lightning.pytorch.strategies import DDPStrategy

torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)  # if using multi-GPU

# model definition
# training loop

if __name__ == '__main__':
    """
    Main training script.
    """
    t = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None, help='Path to config file')
    parser.add_argument('--fold', type=str, default=None, help='Fold number')
    parser.add_argument('--max_epochs', type=int, default=100, 
                        help='Maximum number of training epochs (default: 100)')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='Number of classes for the task (default: 1)')
    parser.add_argument('--accumulate_grad_batches', type=int, default=32, 
                        help='Number of batches to accumulate gradients over (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay for the optimizer (default: 1e-3)')
    parser.add_argument('--input_dim', type=int, default=2560,
                        help='Input dimension size for the MIL model (default: 2560)')
    parser.add_argument('--hidden_dim', type=int, default=1280,
                        help='Hidden dimension size for the MIL model (default: 1280)')
    parser.add_argument('--output_dim', type=int, default=256,
                        help='Common final dimension for all MIL Experts (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate for the MIL model (default: 0.25)')
    parser.add_argument('--comment', type=str, default='',
                        help='Comment to add to the experiment name (default: empty)')
    parser.add_argument('--use_discriminator', type=bool, default=True,
                        help='Whether to use the discriminator (default: True)')
    
    args = parser.parse_args()

    # read config file
    with open(args.config_file, "r") as file_yml:
        config_file = yaml.safe_load(file_yml)

    folds_dir = Path(config_file['train_MIL_discriminator']['folds_dir'])
    hierarchical_h5_dir = Path(config_file['train_MIL_discriminator']['hierarchical_h5_dir'])
    outdir = Path(config_file['train_MIL_discriminator']['out_dir'])
    moe_args = config_file['model'].get('moe_args', {})
    
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    
    fold_num=args.fold
    # fold_num=0# For now, just use fold 0; can be parameterized later
    train_split_file = folds_dir / f'fold_{fold_num}_predictions.csv'

    if args.use_discriminator:
        use_disc = True
    else:
        use_disc = False

    # create dataset and dataloader
    train_dataset = MILDataset(
        csv_path=train_split_file,
        representation_dir=hierarchical_h5_dir,
        max_tiles=None,
        split='train',
        use_discriminator=use_disc,
    )
    
    val_dataset = MILDataset(
        csv_path=train_split_file,
        representation_dir=hierarchical_h5_dir,
        max_tiles=None,
        split='val',
        use_discriminator=use_disc,
    )
    collate_fn = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    pos_weight = train_dataset.get_pos_weight()

    # Load the MIL model
    model = MammothTrainer(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        n_classes=args.n_classes,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        moe_args=moe_args,
        pos_weight=pos_weight,
    )
    
# Create callbacks
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=outdir,
    #     monitor="val_AUROC",
    #     save_top_k=1,
    #     mode="max",
    #     filename=f"fold_{fold_num}" + "-{epoch:02d}-{val_AUROC:.2f}" + f"-{args.comment}")
    
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=outdir,
    #     monitor="val_accuracy",
    #     save_top_k=1,
    #     mode="max",
    #     filename=f"fold_{fold_num}" + "-{epoch:02d}-{val_accuracy:.2f}" + f"-{args.comment}")
    
    early_stopping_callback = EarlyStopping(
        monitor="val_AUROC",
        patience=100, 
        mode="max",
        min_delta=0.002
    )

    # logger = TensorBoardLogger(outdir / "tb_logs", name=f"fold_{fold_num}_{args.model_type}_{args.comment}")
    # logger = WandbLogger(project="Discriminator", name=f"fold_{fold_num}_{args.comment}", 
    #                      save_dir=outdir, log_model=True)

    # Create the trainer
    trainer = L.Trainer(
        # logger=logger,
        max_epochs=args.max_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator='gpu',
        limit_train_batches=0.3, limit_val_batches=0.1,
        devices=1,)
    
    #limit_train_batches=0.1, limit_val_batches=0.01

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )

    # # Check best model's metrics
    # best_path = checkpoint_callback.best_model_path
    # checkpoint  = torch.load(best_path, map_location='cpu')

    # metrics = {
    #     "Validation Loss":     checkpoint['val_loss'],
    #     "Validation Accuracy": checkpoint['val_accuracy'],
    #     "Validation F1":       checkpoint['val_F1'],
    #     "Validation AUROC":    checkpoint['val_AUROC'],
    # }

    # col_w = 22
    # print("┌" + "─" * col_w + "┬" + "─" * 12 + "┐")
    # print(f"│ {'Metric':<{col_w - 2}} │ {'Value':<10} │")
    # print("├" + "─" * col_w + "┼" + "─" * 12 + "┤")
    # for name, val in metrics.items():
    #     print(f"│ {name:<{col_w - 2}} │ {val:<10.4f} │")
    # print("└" + "─" * col_w + "┴" + "─" * 12 + "┘")