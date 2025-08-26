import torch
from torch.utils.data import DataLoader
from torch import optim
import torchvision.transforms as T

from utils.get_model import get_model
from utils.dataset import HumanDataset
from utils.losses import BCEDiceLoss, DiceLoss, BCELoss
from utils.metrics import dice_coef, iou_score
from utils.trainer import Trainer

import argparse
import os

def main(args):
    img_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = T.ToTensor()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset and DataLoader

    train_dataset = HumanDataset(
        images_dir=os.path.join(args.data_dir, "train/images"),
        masks_dir=os.path.join(args.data_dir, "train/masks"),
        img_transform=img_transform,
        mask_transform=mask_transform,
    )   
    
    val_dataset = HumanDataset(
        images_dir=os.path.join(args.data_dir, "val/images"),
        masks_dir=os.path.join(args.data_dir, "val/masks"),
        img_transform=img_transform,
        mask_transform=mask_transform,
    )   
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Model
    model = get_model(model_name="unet", num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    loss_fn = BCEDiceLoss()
    if args.loss_fn.lower() == "bce":
        loss_fn = BCELoss()
    elif args.loss_fn.lower() == "dice":
        loss_fn = DiceLoss()
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) if args.scheduler else None

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        metric_dict={"dice": dice_coef, "iou": iou_score},
        checkpoint_dir=args.checkpoint_dir,
        scheduler=scheduler
    )
    
    if args.checkpoint_path:
        trainer.load_checkpoint(args.checkpoint_path)
    
    trainer.train(num_epochs=args.num_epochs)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to resume checkpoint")
    parser.add_argument("--num_epochs", type=int, default=100, help="'Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss_fn', type=str, default='bce_dice', choices=['bce_dice', 'bce', 'dice'], help='Loss function to use')
    parser.add_argument("--scheduler", type=bool, default=False, help="Learning rate scheduler")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    
    args = parser.parse_args()
    main(args)
