"""Evolution-Weighted 3D Dice + Focal segmentation loss for longitudinal t2 lesion prediction."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


class EvolutionWeightedDiceFocalLoss(nn.Module):
    """Combines soft 3D Dice Loss with a voxel-weighted Focal Loss emphasizing evolving lesions.

    Voxels that evolved between t1 and t2 (|M_t2 - M_t1| > 0.5) receive a weight multiplier
    `change_voxel_weight` (default 2.0) so the decoder prioritizes new/enlarging and shrinking
    lesion boundaries rather than only static chronic core voxels.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        change_voxel_weight: float = 2.0,
        gamma: float = 2.0,
        alpha: float = 0.75,
    ) -> None:
        super().__init__()
        self.dice_loss_fn = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.change_voxel_weight = float(change_voxel_weight)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def _weighted_binary_focal_loss(
        self,
        logits: torch.Tensor,
        target_t2: torch.Tensor,
        target_t1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target_t2, reduction="none")
        p_t = probs * target_t2 + (1.0 - probs) * (1.0 - target_t2)
        alpha_t = self.alpha * target_t2 + (1.0 - self.alpha) * (1.0 - target_t2)
        focal = alpha_t * ((1.0 - p_t).clamp(min=1e-6) ** self.gamma) * bce

        if target_t1 is not None and self.change_voxel_weight > 1.0:
            change_mask = (torch.abs(target_t2 - target_t1) > 0.5).float()
            voxel_weights = 1.0 + (self.change_voxel_weight - 1.0) * change_mask
            focal = focal * voxel_weights

        return focal.mean()

    def forward(
        self,
        logits_t2: torch.Tensor,
        mask_t2: torch.Tensor,
        mask_t1: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        dice_loss = self.dice_loss_fn(logits_t2, mask_t2)
        focal_loss = self._weighted_binary_focal_loss(logits_t2, mask_t2, mask_t1)
        total = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return {
            "loss": total,
            "dice_loss": dice_loss,
            "focal_loss": focal_loss,
        }
