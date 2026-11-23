"""A100-optimized mixed-precision training engines for Phase B SimCLR and Downstream Segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from src.losses.contrastive_loss import HybridLongitudinalSSLLoss
from src.losses.segmentation_loss import EvolutionWeightedDiceFocalLoss
from src.models.segmentation_decoder import SiameseLongitudinalUNet3D
from src.models.simclr import LongitudinalSimCLR3D
from src.utils.config import get_amp_dtype


def train_simclr_epoch(
    model: LongitudinalSimCLR3D,
    dataloader: DataLoader,
    criterion: HybridLongitudinalSSLLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype_str: str = "bfloat16",
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one epoch of Phase B Hybrid Longitudinal SimCLR pretraining."""
    model.train()
    amp_enabled = use_amp and (device.type == "cuda")
    amp_dtype = get_amp_dtype(amp_dtype_str) if amp_enabled else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    totals = {"loss": 0.0, "nt_xent_loss": 0.0, "delta_t_loss": 0.0, "top1_acc": 0.0}
    n_batches = 0

    for batch in dataloader:
        p_t1 = batch["patch_t1"].to(device, non_blocking=True)
        p_t2 = batch["patch_t2"].to(device, non_blocking=True)
        delta_t = batch["delta_t"].to(device, non_blocking=True)
        patient_label = batch["patient_label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(p_t1, p_t2)
            loss_dict = criterion(
                proj_t1=out["proj_t1"],
                proj_t2=out["proj_t2"],
                pred_delta_t=out["pred_delta_t"],
                target_delta_t=delta_t,
                patient_labels=patient_label,
            )
            loss = loss_dict["loss"]

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k in totals:
            totals[k] += float(loss_dict[k].detach().item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def fit_longitudinal_simclr(
    model: LongitudinalSimCLR3D,
    dataloader: DataLoader,
    epochs: int = 40,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    lambda_delta_t: float = 0.25,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    amp_dtype_str: str = "bfloat16",
    save_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, float]]:
    """Train Phase B Longitudinal SimCLR and optionally save adapted backbone weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = HybridLongitudinalSSLLoss(temperature=temperature, lambda_delta_t=lambda_delta_t)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        metrics = train_simclr_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            amp_dtype_str=amp_dtype_str,
        )
        scheduler.step()
        metrics["epoch"] = float(epoch)
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(metrics)

    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.backbone.state_dict(), save_p)

    return history


def train_segmentation_epoch(
    model: SiameseLongitudinalUNet3D,
    dataloader: DataLoader,
    criterion: EvolutionWeightedDiceFocalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype_str: str = "bfloat16",
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one epoch of supervised longitudinal t2 mask segmentation training."""
    model.train()
    # Keep frozen backbone BatchNorm layers in eval mode so running stats do not drift
    if not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()

    amp_enabled = use_amp and (device.type == "cuda")
    amp_dtype = get_amp_dtype(amp_dtype_str) if amp_enabled else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    totals = {"loss": 0.0, "dice_loss": 0.0, "focal_loss": 0.0, "patch_dice": 0.0}
    n_batches = 0

    for batch in dataloader:
        p_t1 = batch["patch_t1"].to(device, non_blocking=True)
        p_t2 = batch["patch_t2"].to(device, non_blocking=True)
        m_t1 = batch["mask_t1"].to(device, non_blocking=True)
        m_t2 = batch["mask_t2"].to(device, non_blocking=True)
        delta_t = batch["delta_t"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits_t2 = model(p_t1, p_t2, delta_t)
            loss_dict = criterion(logits_t2, m_t2, mask_t1=m_t1)
            loss = loss_dict["loss"]

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits_t2) > 0.5).float()
            inter = (preds * m_t2).sum(dim=(1, 2, 3, 4))
            denom = preds.sum(dim=(1, 2, 3, 4)) + m_t2.sum(dim=(1, 2, 3, 4))
            batch_dice = ((2.0 * inter + 1e-5) / (denom + 1e-5)).mean().item()

        totals["loss"] += float(loss_dict["loss"].detach().item())
        totals["dice_loss"] += float(loss_dict["dice_loss"].detach().item())
        totals["focal_loss"] += float(loss_dict["focal_loss"].detach().item())
        totals["patch_dice"] += float(batch_dice)
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}
