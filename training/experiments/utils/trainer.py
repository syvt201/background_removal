import torch
import os
import csv
from tqdm import tqdm, trange

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device, metric_dict, checkpoint_dir="checkpoints/", scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.metric_dict = metric_dict # {"dice": func, "iou": func}
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        
    def init_logger(self):
        self.log_file = os.path.join(self.checkpoint_dir, "training_log.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'dice', 'iou'])
    
    def log_epoch(self, epoch, train_loss, val_loss, val_metrics):
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                val_loss,
                val_metrics.get("dice", 0),
                val_metrics.get("iou", 0)
            ])
            
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss
        }   
        if epoch % 20 == 0:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch:04d}.pth"))
            
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_last_epoch.pth"))
        
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"best_model.pth"))
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from {checkpoint_path}, start from epoch {self.start_epoch}")
        
    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        
        for imgs, masks in loop:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            
            outputs = self.model(imgs)
            loss = self.loss_fn(outputs, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        metrics_total = {name : 0 for name in self.metric_dict.keys()}
        
        with torch.inference_mode():
            loop = tqdm(self.val_loader, desc="Validation", leave=False)
            for imgs, masks in loop:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(imgs)
                loss = self.loss_fn(outputs, masks)
                val_loss += loss.item()
                
                for name, metric_fn in self.metric_dict.items():
                    metrics_total[name] += metric_fn(outputs, masks)

                    
        avg_metrics = {name : total / len(self.val_loader) for name, total in metrics_total.items()}    
        
        return val_loss / len(self.val_loader), avg_metrics
        
    def train(self, num_epochs):
        self.init_logger()
        # epoch_bar = trange(self.start_epoch, num_epochs, desc="Epochs", unit="epoch")
        epoch_bar = trange(self.start_epoch, num_epochs, desc="Epochs", unit="epoch", initial=self.start_epoch, total=num_epochs)
        for epoch in epoch_bar:
            train_loss = self.train_one_epoch()
            val_loss, val_metrics = self.validate()
            
            if self.scheduler: 
                self.scheduler.step(val_loss)
            
            print(f"Epoch [{epoch + 1} / {num_epochs}]:", 
                  f"     Train loss: {train_loss:4f}   Val loss: {val_loss:.4f}  Metrics: {val_metrics}", 
                  sep="")
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            self.log_epoch(epoch, train_loss, val_loss, val_metrics)
            
            
        