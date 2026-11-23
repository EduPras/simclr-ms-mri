"""Siamese Longitudinal 3D U-Net Decoder with multi-scale feature differencing and FiLM(Δt)."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import MedicalNet3DResNet18


class TemporalDifferenceFusionBlock3D(nn.Module):
    """Fuses [F_t2, F_t2 - F_t1, |F_t2 - F_t1|] at pyramid scale l into C_out channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_t1: torch.Tensor, f_t2: torch.Tensor) -> torch.Tensor:
        diff = f_t2 - f_t1
        concat = torch.cat([f_t2, diff, torch.abs(diff)], dim=1)
        return self.proj(concat)


class TimeFiLM3D(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioning bottleneck features on scalar Δt."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, channels * 2),
        )

    def forward(self, feat: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        if delta_t.ndim == 1:
            delta_t = delta_t.unsqueeze(-1)
        params = self.mlp(delta_t)  # [B, 2*C]
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.view(feat.shape[0], feat.shape[1], 1, 1, 1)
        beta = beta.view(feat.shape[0], feat.shape[1], 1, 1, 1)
        return (1.0 + gamma) * feat + beta


class DecoderUpBlock3D(nn.Module):
    """3D upsampling block that dynamically aligns to skip resolution and applies residual 3D convs."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SiameseLongitudinalUNet3D(nn.Module):
    """Siamese 3D MedicalNet + Multi-Scale Temporal Difference & FiLM(Δt) Decoder for t2 Mask Prediction."""

    def __init__(
        self,
        backbone: MedicalNet3DResNet18,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            self.backbone.freeze_all()

        c0, c1, c2, c3, c4 = 64, 64, 128, 256, 512
        d3_ch, d2_ch, d1_ch, d0_ch = decoder_channels

        # Multi-scale temporal difference fusion blocks
        self.fuse0 = TemporalDifferenceFusionBlock3D(c0, d0_ch)
        self.fuse1 = TemporalDifferenceFusionBlock3D(c1, d1_ch)
        self.fuse2 = TemporalDifferenceFusionBlock3D(c2, d2_ch)
        self.fuse3 = TemporalDifferenceFusionBlock3D(c3, d3_ch)
        self.fuse4 = TemporalDifferenceFusionBlock3D(c4, d3_ch)

        # Bottleneck Δt FiLM modulation
        self.film_bottleneck = TimeFiLM3D(d3_ch)

        # Progressive 3D decoder path
        self.up3 = DecoderUpBlock3D(d3_ch, d3_ch, d2_ch)
        self.up2 = DecoderUpBlock3D(d2_ch, d2_ch, d1_ch)
        self.up1 = DecoderUpBlock3D(d1_ch, d1_ch, d0_ch)
        self.up0 = DecoderUpBlock3D(d0_ch, d0_ch, d0_ch)

        # Raw input skip refinement (from 4-channel x_t2 and x_t2 - x_t1 at full resolution)
        self.full_res_head = nn.Sequential(
            nn.Conv3d(d0_ch + self.backbone.in_channels * 2, d0_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(d0_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(d0_ch, 1, kernel_size=1),
        )

    def forward_with_intermediates(
        self,
        x_t1: torch.Tensor,
        x_t2: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run forward pass returning predicted t2 logits and multi-scale feature difference maps."""
        feats_t1 = self.backbone.forward_features(x_t1)
        feats_t2 = self.backbone.forward_features(x_t2)

        d0 = self.fuse0(feats_t1["f0"], feats_t2["f0"])
        d1 = self.fuse1(feats_t1["f1"], feats_t2["f1"])
        d2 = self.fuse2(feats_t1["f2"], feats_t2["f2"])
        d3 = self.fuse3(feats_t1["f3"], feats_t2["f3"])
        d4 = self.fuse4(feats_t1["f4"], feats_t2["f4"])

        b = self.film_bottleneck(d4, delta_t)

        u3 = self.up3(b, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)
        u0 = self.up0(u1, d0)

        if u0.shape[2:] != x_t2.shape[2:]:
            u0 = F.interpolate(u0, size=x_t2.shape[2:], mode="trilinear", align_corners=False)

        raw_skip = torch.cat([x_t2, x_t2 - x_t1], dim=1)
        logits_t2 = self.full_res_head(torch.cat([u0, raw_skip], dim=1))

        # L2 norm of deep feature difference for interpretability visualization
        f1_diff_norm = torch.norm(feats_t2["f1"] - feats_t1["f1"], p=2, dim=1, keepdim=True)

        return {
            "logits_t2": logits_t2,
            "feature_diff_norm": f1_diff_norm,
            "bottleneck_diff": b,
        }

    def forward(
        self,
        x_t1: torch.Tensor,
        x_t2: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict t2 lesion segmentation logits [B, 1, H, W, D]."""
        return self.forward_with_intermediates(x_t1, x_t2, delta_t)["logits_t2"]

    def forward_stacked_pair(
        self,
        x_stacked: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        """Helper for MONAI sliding_window_inference where x_t1 and x_t2 are stacked along channel axis."""
        c = self.backbone.in_channels
        x_t1 = x_stacked[:, :c]
        x_t2 = x_stacked[:, c : 2 * c]
        if delta_t.shape[0] != x_stacked.shape[0]:
            delta_t = delta_t.expand(x_stacked.shape[0], -1)
        return self.forward(x_t1, x_t2, delta_t)
