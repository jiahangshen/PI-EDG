"""Train the paper-version PI-EDG model on cached EDBench density grids."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from dataset_moe_phys import PhysicsMoEDataset, collate_fn_moe_phys, load_sample_ids
from model_softmoe_phys import PhysicsMoE, normalize_density_to_electrons
from physics_utils import PhysicsUtils


def build_initial_bins():
    return [0.0] + [10.0**exponent for exponent in range(-34, 2)] + [float("inf")]


class MoEPhysModule(pl.LightningModule):
    def __init__(
        self,
        grid_res=64,
        grid_bounds=(-10.0, 10.0),
        learning_rate=1e-5,
        guide_weight=1e-5,
        guide_mask_threshold=1e-10,
        routing_backbone_grad_scale=1.0,
        threshold_lr_scale=0.1,
        voxel_volume_bohr3=0.064,
        num_pe_freqs=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        initial_bins = build_initial_bins()
        self.model = PhysicsMoE(
            in_channels_atom=13,
            in_channels_pe=3 * 2 * num_pe_freqs,
            num_experts=len(initial_bins) - 1,
            initial_bins=initial_bins,
            routing_backbone_grad_scale=routing_backbone_grad_scale,
        )
        self.register_buffer(
            "grid_buffer",
            PhysicsUtils.get_grid_coords(grid_res, grid_bounds, device="cpu"),
        )

    def forward(self, atom_grid, pe_grid, v_ext, z_spike):
        return self.model(atom_grid, pe_grid, v_ext, z_spike)

    def _step(self, batch, stage):
        atom_grid = batch["atom_grid"]
        target = batch["target_density"]
        with torch.no_grad():
            v_ext = PhysicsUtils.build_vext(
                batch["atom_coords"], batch["atom_z"], self.grid_buffer
            )
        _, _, pred, guide = self(
            atom_grid,
            batch["pe_grid"],
            v_ext,
            atom_grid.sum(dim=1, keepdim=True),
        )

        pred_cubic = torch.pow(pred + 1e-8, 1.0 / 3.0)
        target_cubic = torch.pow(target + 1e-8, 1.0 / 3.0)
        density_loss = F.mse_loss(pred_cubic, target_cubic)
        guide_mask = target > self.hparams.guide_mask_threshold
        if bool(guide_mask.any()):
            guide_loss = F.mse_loss(
                guide[guide_mask], torch.log10(target[guide_mask] + 1e-35)
            )
        else:
            guide_loss = density_loss.new_zeros(())
        loss = density_loss + self.hparams.guide_weight * guide_loss
        mae = F.l1_loss(pred, target)
        batch_size = atom_grid.shape[0]
        self.log(f"{stage}_loss", loss, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_mae", mae, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        if stage == "val":
            nominal = normalize_density_to_electrons(
                pred.float(),
                batch["target_electrons"].float(),
                self.hparams.voxel_volume_bohr3,
            )
            self.log("val_normalized_mae", F.l1_loss(nominal, target), batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        threshold = self.model.router.log_gaps
        other_params = [param for param in self.parameters() if param is not threshold]
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": self.hparams.learning_rate},
                {"params": [threshold], "lr": self.hparams.learning_rate * self.hparams.threshold_lr_scale},
            ],
            foreach=False,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def load_model_weights(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = MoEPhysModule()
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, default=Path("splits/train_ids.npy"))
    parser.add_argument("--val-ids", type=Path, default=Path("splits/validation_ids.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pi_edg"))
    parser.add_argument("--devices", type=str, default="0", help="Comma-separated GPU indices")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    train_ids = load_sample_ids(args.train_ids)
    val_ids = load_sample_ids(args.val_ids)
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and validation IDs overlap")
    pl.seed_everything(args.seed, workers=True)
    train_set = PhysicsMoEDataset(args.cache_dir, train_ids)
    val_set = PhysicsMoEDataset(args.cache_dir, val_ids)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn_moe_phys,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    model = MoEPhysModule(learning_rate=args.learning_rate)
    devices = [int(value) for value in args.devices.split(",")]
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_mae:.6f}",
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        max_epochs=args.epochs,
        precision=args.precision,
        logger=CSVLogger(args.output_dir, name="logs"),
        callbacks=[checkpoint],
        gradient_clip_val=1.0,
        strategy="ddp" if len(devices) > 1 else "auto",
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(args.resume) if args.resume else None)


if __name__ == "__main__":
    main()
