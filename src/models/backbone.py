"""MONAI 3D MedicalNet ResNet-18 backbone with 4-channel weight inflation and phase freeze control."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet18

logger = logging.getLogger(__name__)


def inflate_conv3d_from_1ch(conv1_1ch: nn.Conv3d, in_channels: int = 4) -> nn.Conv3d:
    """Inflate a pretrained 1-channel 3D convolution kernel to `in_channels` via mean replication.

    Preserves activation scale: if a 1-channel volume is repeated across `in_channels`,
    the output of the inflated 4-channel conv matches the 1-channel conv output exactly.
    """
    if in_channels == 1:
        return conv1_1ch

    new_conv = nn.Conv3d(
        in_channels=in_channels,
        out_channels=conv1_1ch.out_channels,
        kernel_size=conv1_1ch.kernel_size,
        stride=conv1_1ch.stride,
        padding=conv1_1ch.padding,
        dilation=conv1_1ch.dilation,
        groups=conv1_1ch.groups,
        bias=(conv1_1ch.bias is not None),
        padding_mode=conv1_1ch.padding_mode,
    )

    with torch.no_grad():
        inflated_weight = conv1_1ch.weight.data.repeat(1, in_channels, 1, 1, 1) / float(in_channels)
        new_conv.weight.data.copy_(inflated_weight)
        if conv1_1ch.bias is not None and new_conv.bias is not None:
            new_conv.bias.data.copy_(conv1_1ch.bias.data)

    return new_conv


class MedicalNet3DResNet18(nn.Module):
    """Multi-scale 3D ResNet-18 backbone initialized from MONAI MedicalNet (Med3D 23-datasets)."""

    FEATURE_CHANNELS = {
        "f0": 64,
        "f1": 64,
        "f2": 128,
        "f3": 256,
        "f4": 512,
    }

    def __init__(
        self,
        in_channels: int = 4,
        pretrained: bool = True,
        allow_offline_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.pretrained_loaded = False

        if pretrained:
            try:
                base = resnet18(
                    pretrained=True,
                    spatial_dims=3,
                    n_input_channels=1,
                    feed_forward=False,
                    shortcut_type="A",
                    bias_downsample=False,
                )
                self.pretrained_loaded = True
            except Exception as exc:
                if not allow_offline_fallback:
                    raise
                logger.warning(
                    "Could not download MedicalNet pretrained weights (%s); falling back to random init.",
                    exc,
                )
                base = resnet18(
                    pretrained=False,
                    spatial_dims=3,
                    n_input_channels=1,
                    feed_forward=False,
                    shortcut_type="A",
                    bias_downsample=False,
                )
        else:
            base = resnet18(
                pretrained=False,
                spatial_dims=3,
                n_input_channels=1,
                feed_forward=False,
                shortcut_type="A",
                bias_downsample=False,
            )

        base.conv1 = inflate_conv3d_from_1ch(base.conv1, in_channels=in_channels)
        self.backbone = base
        self._act = getattr(self.backbone, "act", getattr(self.backbone, "relu", nn.ReLU(inplace=True)))

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract hierarchical 3D feature pyramid [f0..f4] and globally pooled embedding."""
        f0 = self._act(self.backbone.bn1(self.backbone.conv1(x)))  # [B, 64, H/2, W/2, D/2]
        p0 = self.backbone.maxpool(f0)                             # [B, 64, H/4, W/4, D/4]
        f1 = self.backbone.layer1(p0)                              # [B, 64, H/4, W/4, D/4]
        f2 = self.backbone.layer2(f1)                              # [B, 128, ...]
        f3 = self.backbone.layer3(f2)                              # [B, 256, ...]
        f4 = self.backbone.layer4(f3)                              # [B, 512, ...]
        pooled = F.adaptive_avg_pool3d(f4, 1).flatten(1)           # [B, 512]

        return {
            "f0": f0,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "pooled": pooled,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return globally pooled 512-D 3D representation."""
        return self.forward_features(x)["pooled"]

    def freeze_all(self) -> None:
        """Freeze 100% of backbone parameters (Phase A Frozen Baseline)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def configure_phase_b_unfreezing(
        self,
        unfreeze_layers: Sequence[str] = ("conv1", "bn1", "layer4"),
    ) -> None:
        """Freeze early anatomical edge layers (layer1-3) and unfreeze specified layers for Phase B."""
        self.freeze_all()
        for layer_name in unfreeze_layers:
            module = getattr(self.backbone, layer_name, None)
            if module is None:
                raise ValueError(f"Backbone has no submodule named '{layer_name}'.")
            for param in module.parameters():
                param.requires_grad = True

    def count_trainable_parameters(self) -> Dict[str, int]:
        """Return counts of trainable and total parameters in the backbone."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable, "total": total}
