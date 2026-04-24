# file: train_classifier.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from tqdm import tqdm
from model import UNet3D 
from dataset_classifier import VoxelizedClassifierDataset, collate_fn_skip_error
from torch.optim import lr_scheduler 
torch.set_float32_matmul_precision('high')
def calculate_class_weights_from_dataset(dataset, num_classes):
    """
    扫描数据集的子集（这里是训练集）来计算类别权重。
    
    参数:
    - dataset: 一个 torch.utils.data.Dataset 对象 (例如，你的 train_dataset)。
    - num_classes: 类别总数。

    返回:
    - class_weights: 一个 torch.Tensor，包含每个类别的权重。
    """
    print(f"开始扫描数据集以计算 {num_classes} 个类别的权重...")
    
    # 使用 DataLoader 来利用多进程加速数据加载
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8, collate_fn=collate_fn_skip_error)
    
    # 初始化计数器
    class_counts = torch.zeros(num_classes, dtype=torch.int64)

    for batch in tqdm(loader, desc="扫描数据集中"):
        # collate_fn 可能会返回空批次，跳过它们
        if not batch:
            continue
            
        _, _,_, target_labels,_,_ = batch
        
        # target_labels 的形状是 [batch, res, res, res]
        # 使用 torch.bincount 高效统计
        # minlength 确保即使批次中没有某个类，结果张量长度也正确
        counts = torch.bincount(target_labels.flatten(), minlength=num_classes)
        class_counts += counts.cpu() # 累加到 CPU 张量上

    print("数据集中各类别的体素总数:", class_counts.numpy())

    # --- 计算权重 ---
    # 权重 = 1 / 频率
    # 加上一个很小的 epsilon 防止除以零
    epsilon = 1e-6
    weights = 1.0 / (class_counts.float() + epsilon)
    
    # --- 归一化权重 ---
    # 这种归一化方式（使其和为类别数）可以使权重的平均值为1，有助于稳定训练
    weights = weights / weights.sum() * num_classes
    
    print("计算得到的类别权重:", weights)
    return weights

class DensityClassifierModel(pl.LightningModule):
    def __init__(self, in_channels_atom, in_channels_pe, num_classes, learning_rate=1e-4, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = UNet3D(
            in_channels_atom=in_channels_atom,
            in_channels_pe=in_channels_pe,
            out_channels=num_classes,
            n_channels=64
        )
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, average=None) 
        
        self.val_iou_macro = MulticlassJaccardIndex(num_classes=num_classes, average='macro')
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')

    def forward(self, atom_grid, pe_grid):
        return self.model(atom_grid, pe_grid)

    def training_step(self, batch, batch_idx):
        atom_grids, pe_grids, target_labels = batch
        if atom_grids.nelement() == 0: return None        
        
        pe_grids = pe_grids.to(torch.float32)
        
        logits = self.forward(atom_grids, pe_grids)
        
        loss = self.criterion(logits, target_labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        atom_grids, pe_grids, target_labels = batch
        if atom_grids.nelement() == 0: return

        pe_grids = pe_grids.to(torch.float32)
        
        logits = self.forward(atom_grids, pe_grids)
        loss = self.criterion(logits, target_labels)
        
        preds = torch.argmax(logits, dim=1)
        
        self.val_iou.update(preds, target_labels)
        self.val_iou_macro.update(preds, target_labels)
        self.val_accuracy.update(preds, target_labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        epoch_iou_macro = self.val_iou_macro.compute()
        epoch_accuracy = self.val_accuracy.compute()
        
        self.log('val_epoch_iou_macro', epoch_iou_macro, prog_bar=True) 
        self.log('val_epoch_accuracy', epoch_accuracy, prog_bar=True)
        
        per_class_iou = self.val_iou.compute()
        
        print("\n" + "="*50)
        print(f"Epoch {self.current_epoch} Validation - Per-Class IoU Report:")
        print("-" * 50)
        for i, iou in enumerate(per_class_iou):
            self.logger.log_metrics({f'val_iou_class_{i}': iou.item()}, step=self.global_step)
            print(f"  - Class {i:<2}: IoU = {iou:.4f}")
        print("-" * 50)
        print(f"  Macro Average IoU: {epoch_iou_macro:.4f}")
        print("=" * 50 + "\n")

        self.val_iou.reset()
        self.val_iou_macro.reset()
        self.val_accuracy.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches, # <-- 关键
            eta_min=1e-6 # 学习率的下限
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1,
                "monitor": "val/total_loss", 
            },
        }
import numpy as np

def main():
    RAW_DATA_ROOT = Path("./raw_data_cube") 
    CACHE_DIR = Path("./preprocessed_cache")
    GRID_RESOLUTION = 64
    GRID_BOUNDS = (-10.0, 10.0)
    NUM_PE_FREQS = 4
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    VAL_SPLIT_RATIO = 0.1
    NUM_WORKERS = 12

    CLASS_BINS = [0, 1e-14,1e-10,1e-7,1e-5, 1e-4, 1e-3, 1e-2, 1.0, 10.0, 100.0, np.inf]
    NUM_CLASSES = len(CLASS_BINS)
    print(f"将电子密度分为 {NUM_CLASSES} 个类别。")

    full_dataset = VoxelizedClassifierDataset(
        cache_dir=CACHE_DIR,
        raw_data_root_for_scan=RAW_DATA_ROOT,
        grid_res=GRID_RESOLUTION,
        grid_bounds=GRID_BOUNDS,
        num_pe_freqs=NUM_PE_FREQS,
        class_bins=CLASS_BINS 
    )
    
    NUM_ATOM_TYPES = full_dataset.num_atom_channels
    NUM_PE_CHANNELS = 3 * 2 * NUM_PE_FREQS
    
    n_total = len(full_dataset)
    n_val = int(n_total * VAL_SPLIT_RATIO)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    

    #class_weights = calculate_class_weights_from_dataset(train_dataset, NUM_CLASSES)
    class_weights=torch.tensor([6.0300e-06, 3.4428e-04, 2.8687e-04, 1.9065e-04, 1.6612e-04, 2.7291e-04,
        2.7877e-04, 3.0336e-04, 2.3761e-04, 5.3823e-03, 9.2821e-02, 1.1900e+01])
    # --- 创建 DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_error, persistent_workers=True)

    model = DensityClassifierModel(
        in_channels_atom=NUM_ATOM_TYPES,
        in_channels_pe=NUM_PE_CHANNELS,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE,
        class_weights=class_weights 
    )
    logger = TensorBoardLogger("logs", name="density_classifier")
    checkpoint_callback = ModelCheckpoint(
            monitor='val_epoch_iou_macro', 
            dirpath='checkpoints_classifier', 
            filename='best-model-{epoch:02d}-iou={val_epoch_iou_macro:.4f}', 
            save_top_k=1, 
            mode='max'
        )    
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=NUM_EPOCHS, logger=logger, callbacks=[checkpoint_callback])
    
    print("开始训练密度分类模型...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("分类模型训练完成！")

if __name__ == "__main__":
    pl.seed_everything(42)
    main()