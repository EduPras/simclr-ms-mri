"""Phase B Hybrid Longitudinal SimCLR 3D model with auxiliary temporal difference (Δt) head."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import MedicalNet3DResNet18


class LongitudinalSimCLR3D(nn.Module):
    """3D Longitudinal SimCLR with nonlinear projection head and auxiliary Δt difference head.

    - Projection Head g_phi(z_t): Maps 512-D pooled backbone features to L2-normalized
      128-D contrastive space for Patient-Masked NT-Xent loss.
    - Auxiliary Temporal Head h_psi([z_t2 - z_t1, |z_t2 - z_t1|]): Predicts normalized
      time interval Δt = (t_2 - t_1) / 4.0 to prevent z_t2 - z_t1 from collapsing to zero.
    """

    def __init__(
        self,
        backbone: MedicalNet3DResNet18,
        embed_dim: int = 512,
        proj_hidden_dim: int = 256,
        proj_out_dim: int = 128,
        unfreeze_layers: Sequence[str] = ("conv1", "bn1", "layer4"),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.configure_phase_b_unfreezing(unfreeze_layers)

        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_out_dim, bias=True),
        )

        self.temporal_delta_head = nn.Sequential(
            nn.Linear(embed_dim * 2, proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, 1),
        )

    def forward(
        self,
        patch_t1: torch.Tensor,
        patch_t2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute latent embeddings, L2-normalized projections, and predicted Δt."""
        z_t1 = self.backbone(patch_t1)  # [B, 512]
        z_t2 = self.backbone(patch_t2)  # [B, 512]

        proj_t1 = F.normalize(self.projection_head(z_t1), dim=-1)  # [B, 128]
        proj_t2 = F.normalize(self.projection_head(z_t2), dim=-1)  # [B, 128]

        diff = z_t2 - z_t1
        diff_features = torch.cat([diff, torch.abs(diff)], dim=-1)  # [B, 1024]
        pred_delta_t = self.temporal_delta_head(diff_features)      # [B, 1]

        return {
            "z_t1": z_t1,
            "z_t2": z_t2,
            "proj_t1": proj_t1,
            "proj_t2": proj_t2,
            "pred_delta_t": pred_delta_t,
        }
