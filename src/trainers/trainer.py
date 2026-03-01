"""
Main training loop for transport neural operators.

Supports:
- AMP (automatic mixed precision)
- Gradient clipping
- EMA (exponential moving average) of model weights
- Tensorboard logging
- Checkpointing with resume
- Deterministic seeds
- Configurable optimizers and schedulers
"""

from __future__ import annotations
import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader

from ..data.dataset import collate_fn, resample_omega_directions

logger = logging.getLogger(__name__)


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (self.decay * self.shadow[name]
                                     + (1 - self.decay) * param.data)

    def apply(self):
        """Apply EMA weights to model (for eval)."""
        self._orig = {}
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self._orig[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original weights (after eval)."""
        for name, param in self.model.named_parameters():
            if name in self._orig:
                param.data.copy_(self._orig[name])
        self._orig = {}


class Trainer:
    """
    Trainer for transport neural operators.

    Handles:
    - Training loop with loss computation
    - Validation
    - Checkpointing and resume
    - AMP, EMA, grad clipping
    - Tensorboard logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        # Optimizer
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        # Schedule
        scheduler: str = "cosine",
        n_epochs: int = 100,
        warmup_epochs: int = 5,
        # Training
        batch_size: int = 8,
        grad_clip: float = 1.0,
        use_amp: bool = False,
        # Angular augmentation
        augment_omega: bool = True,
        # EMA
        use_ema: bool = False,
        ema_decay: float = 0.999,
        # Logging
        log_dir: str = "runs",
        run_name: str = "run",
        log_every: int = 10,
        val_every: int = 1,
        # Resume
        checkpoint_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        # Device
        device: Optional[str] = None,
        # Workers
        num_workers: int = 0,
        seed: int = 42,
        # Model hyperparameters (stored in checkpoint for reproducible eval)
        model_args: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.log_every = log_every
        self.val_every = val_every
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.augment_omega = augment_omega
        self.seed = seed
        self.model_args = model_args or {}
        self._aug_rng = np.random.default_rng(seed + 1)

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler
        total_steps = n_epochs * len(self.train_loader)
        warmup_steps = warmup_epochs * len(self.train_loader)
        if scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=lr * 0.01,
            )
        elif scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                total_steps=total_steps,
            )
        else:
            self.scheduler = None

        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None

        # EMA
        self.ema = EMA(model, ema_decay) if use_ema else None

        # Logging
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
        except ImportError:
            self.writer = None
            logger.warning("tensorboard not available; skipping TB logging.")

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        if resume_from:
            self._load_checkpoint(resume_from)

    def _compute_loss(self, pred: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Dispatch to model-specific loss if available, else use default L2 loss."""
        if hasattr(self.model, "compute_loss"):
            return self.model.compute_loss(pred, batch)

        # Default: relative MSE on I and phi, normalised by the RMS of the
        # *valid* (unmasked) target elements.  Computing RMS over all elements
        # (including zero-padded boundaries/void cells) deflates the scale by
        # ~100× and makes the normalised loss meaningless.
        I_pred = pred["I"]
        I_true = batch["I"].to(self.device)
        omega_mask = batch.get("omega_mask")
        if omega_mask is not None:
            omega_mask = omega_mask.to(self.device)
            mask = omega_mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, Nw, 1]
            n_valid = mask.expand_as(I_true).sum().clamp(min=1)   # B*Nx*Nw_valid*G
            loss_I = ((I_pred - I_true)**2 * mask).sum() / n_valid
            I_scale = ((I_true.detach()**2 * mask).sum() / n_valid).clamp(min=1e-10).sqrt()
        else:
            loss_I = torch.nn.functional.mse_loss(I_pred, I_true)
            I_scale = I_true.detach().pow(2).mean().clamp(min=1e-10).sqrt()
        loss_I = loss_I / (I_scale ** 2)

        phi_pred = pred.get("phi")
        phi_true = batch["phi"].to(self.device)
        if phi_pred is not None:
            phi_scale = phi_true.detach().pow(2).mean().clamp(min=1e-10).sqrt()
            loss_phi = torch.nn.functional.mse_loss(phi_pred, phi_true) / (phi_scale ** 2)
        else:
            loss_phi = torch.tensor(0.0, device=self.device)

        loss_total = loss_I + 0.1 * loss_phi
        return {
            "loss_total": loss_total,
            "loss_intensity": loss_I,
            "loss_phi": loss_phi,
        }

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensor values in batch to device."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _augment_batch_omega(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Angular-direction augmentation on a collated batch (CPU tensors).

        Randomly rotates the quadrature grid by a per-batch offset so that the
        model sees a different set of discrete SN directions every iteration.
        The target I is recomputed via the P1 formula from stored phi/J.

        For C5G7, OpenMC provides only the scalar flux φ(x) via tallies.  The
        angular intensity I(x,ω) stored in the dataset is P1-reconstructed from
        φ and J_Fick at a fixed S8 quadrature (8 uniformly-spaced directions).
        Because I is a P1 field derived from stored φ and J, rotating ω and
        recomputing I from those same φ/J values is exactly self-consistent —
        the augmented targets are as valid as the originals.  This augmentation
        therefore teaches the model discretisation-agnosticism at zero cost.

        Only applied when self.augment_omega is True.
        """
        if not self.augment_omega:
            return batch

        omega = batch.get("omega")   # [B, Nw, dim]
        I     = batch.get("I")       # [B, Nx, Nw, G]
        phi   = batch.get("phi")     # [B, Nx, G]
        J     = batch.get("J")       # [B, Nx, dim, G]

        if omega is None or I is None or phi is None or J is None:
            return batch

        dim = omega.shape[-1]
        n_omega = omega.shape[1]

        # For 2-D problems: apply a random rotation offset to the whole grid.
        if dim == 2:
            offset = float(self._aug_rng.uniform(0, 2 * np.pi / n_omega))
            theta_new = (torch.atan2(omega[..., 1], omega[..., 0]) + offset)  # [B, Nw]
            omega_new = torch.stack([torch.cos(theta_new), torch.sin(theta_new)], dim=-1)

            # Recompute I at the new directions via P1 from stored phi, J
            norm = 2.0 * np.pi
            phi_safe = phi.clamp(min=1e-30).unsqueeze(2)                       # [B, Nx, 1, G]
            JdotOmega = torch.einsum('bndg,bwd->bnwg', J, omega_new)           # [B, Nx, Nw, G]
            correction = (1.0 + (2.0 / norm) * JdotOmega / phi_safe).clamp(min=0.0)
            I_new = (phi.unsqueeze(2) / norm) * correction

            batch = dict(batch)
            batch["omega"] = omega_new
            batch["I"] = I_new

        # 3-D: apply a random rotation about the z-axis (keeps the quadrature uniform)
        elif dim == 3:
            angle = float(self._aug_rng.uniform(0, 2 * np.pi))
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            ox, oy, oz = omega[..., 0], omega[..., 1], omega[..., 2]
            ox_new = cos_a * ox - sin_a * oy
            oy_new = sin_a * ox + cos_a * oy
            omega_new = torch.stack([ox_new, oy_new, oz], dim=-1)

            norm = 4.0 * np.pi
            phi_safe = phi.clamp(min=1e-30).unsqueeze(2)
            JdotOmega = torch.einsum('bndg,bwd->bnwg', J, omega_new)
            correction = (1.0 + (3.0 / norm) * JdotOmega / phi_safe).clamp(min=0.0)
            I_new = (phi.unsqueeze(2) / norm) * correction

            batch = dict(batch)
            batch["omega"] = omega_new
            batch["I"] = I_new

        return batch

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_losses: Dict[str, float] = {}
        n_batches = 0

        for batch in self.train_loader:
            batch = self._move_batch(batch)
            batch = self._augment_batch_omega(batch)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = self.model(batch)
                    losses = self._compute_loss(pred, batch)
                self.scaler.scale(losses["loss_total"]).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(batch)
                losses = self._compute_loss(pred, batch)
                losses["loss_total"].backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.ema is not None:
                self.ema.update()

            # Accumulate losses
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                step_losses = {k: v / n_batches for k, v in total_losses.items()}
                if self.writer:
                    for k, v in step_losses.items():
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)
                lr = self.optimizer.param_groups[0]["lr"]
                if self.writer:
                    self.writer.add_scalar("train/lr", lr, self.global_step)

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        if self.ema is not None:
            self.ema.apply()

        self.model.eval()
        total_losses: Dict[str, float] = {}
        n_batches = 0

        for batch in self.val_loader:
            batch = self._move_batch(batch)
            pred = self.model(batch)
            losses = self._compute_loss(pred, batch)
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            n_batches += 1

        if self.ema is not None:
            self.ema.restore()

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    def train(self):
        """Full training loop."""
        logger.info(f"Starting training: {self.n_epochs} epochs, device={self.device}")
        for epoch in range(self.epoch, self.n_epochs):
            self.epoch = epoch
            t0 = time.time()
            train_losses = self.train_epoch()
            t1 = time.time()

            log_str = f"Epoch {epoch+1}/{self.n_epochs} | time={t1-t0:.1f}s"
            for k, v in train_losses.items():
                log_str += f" | train_{k}={v:.4e}"

            val_losses = {}
            if (epoch + 1) % self.val_every == 0:
                val_losses = self.validate()
                for k, v in val_losses.items():
                    log_str += f" | val_{k}={v:.4e}"
                if self.writer:
                    for k, v in val_losses.items():
                        self.writer.add_scalar(f"val/{k}", v, epoch)

            logger.info(log_str)

            # Save checkpoint
            val_loss = val_losses.get("loss_total", train_losses.get("loss_total", float("inf")))
            self._save_checkpoint(epoch, val_loss)

        if self.writer:
            self.writer.close()
        logger.info("Training complete.")

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save training checkpoint."""
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "model_args": self.model_args,
        }
        if self.scheduler is not None:
            state["scheduler_state"] = self.scheduler.state_dict()
        if self.ema is not None:
            state["ema_shadow"] = self.ema.shadow

        # Always save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(state, latest_path)

        # Save best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            # Update best_val_loss in the saved state before writing
            state["best_val_loss"] = self.best_val_loss
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            logger.info(f"New best model saved: val_loss={val_loss:.4e}")

    def _load_checkpoint(self, path: str):
        """Load training state from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if "ema_shadow" in ckpt and self.ema is not None:
            self.ema.shadow = ckpt["ema_shadow"]
        self.epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from {path}: epoch={self.epoch}")

    def save_model_only(self, path: str):
        """Save only model weights (for deployment)."""
        torch.save(self.model.state_dict(), path)
