"""3D volumetric loading, caching, foreground Z-score normalization, and aligned 3D patch datasets."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.isbi_dataset import DEFAULT_MODALITIES, LongitudinalPairRecord, VisitRecord


def load_nifti_ras(path: str) -> np.ndarray:
    """Load a 3D NIfTI file as a float32 NumPy array."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return np.asarray(data, dtype=np.float32)


def normalize_foreground_zscore(volume_4d: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Per-modality Z-score normalization over the non-zero brain foreground mask.

    Args:
        volume_4d: Array of shape [C, H, W, D].

    Returns:
        norm_vol: Normalized array of shape [C, H, W, D], background zeroed.
        fg_mask: Boolean foreground mask of shape [H, W, D].
    """
    fg_mask = np.any(np.abs(volume_4d) > eps, axis=0)
    norm_vol = np.zeros_like(volume_4d, dtype=np.float32)

    if not np.any(fg_mask):
        return norm_vol, fg_mask

    for c in range(volume_4d.shape[0]):
        ch = volume_4d[c]
        fg_vals = ch[fg_mask]
        mean = float(np.mean(fg_vals))
        std = float(np.std(fg_vals))
        norm_ch = np.zeros_like(ch, dtype=np.float32)
        norm_ch[fg_mask] = (fg_vals - mean) / max(std, eps)
        norm_vol[c] = norm_ch

    return norm_vol, fg_mask


def resolve_target_mask(
    mask1: Optional[np.ndarray],
    mask2: Optional[np.ndarray],
    target_rater: str = "mask1",
) -> Optional[np.ndarray]:
    """Combine or select binary expert rater masks (mask1, mask2, union, intersection)."""
    if mask1 is None and mask2 is None:
        return None
    if target_rater == "mask1" or mask2 is None:
        return (mask1 > 0.5).astype(np.float32) if mask1 is not None else None
    if target_rater == "mask2" or mask1 is None:
        return (mask2 > 0.5).astype(np.float32)
    if target_rater == "union":
        return ((mask1 > 0.5) | (mask2 > 0.5)).astype(np.float32)
    if target_rater == "intersection":
        return ((mask1 > 0.5) & (mask2 > 0.5)).astype(np.float32)
    raise ValueError(f"Unknown target_rater: {target_rater}")


class InMemoryVolumeCache:
    """Caches normalized 4-channel 3D visits in system RAM (ideal for Colab Pro A100 83GB RAM)."""

    def __init__(
        self,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        target_rater: str = "mask1",
        enabled: bool = True,
    ) -> None:
        self.modalities = tuple(m.lower() for m in modalities)
        self.target_rater = target_rater
        self.enabled = enabled
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def get_visit_data(self, visit: VisitRecord) -> Dict[str, np.ndarray]:
        """Load or retrieve cached 4-channel normalized volume, foreground coords, and masks."""
        if self.enabled and visit.visit_id in self._cache:
            return self._cache[visit.visit_id]

        channels = [load_nifti_ras(str(visit.modality_paths[m])) for m in self.modalities]
        vol_4d = np.stack(channels, axis=0)  # [C, H, W, D]
        norm_vol, fg_mask = normalize_foreground_zscore(vol_4d)

        mask1 = load_nifti_ras(str(visit.mask1_path)) if visit.mask1_path else None
        mask2 = load_nifti_ras(str(visit.mask2_path)) if visit.mask2_path else None
        target_mask = resolve_target_mask(mask1, mask2, self.target_rater)

        fg_coords = np.argwhere(fg_mask)
        if len(fg_coords) == 0:
            # Fallback to volume center if synthetic volume is all zeros
            h, w, d = norm_vol.shape[1:]
            fg_coords = np.array([[h // 2, w // 2, d // 2]], dtype=np.int64)
        elif len(fg_coords) > 20000:
            # Subsample foreground candidate centers for fast O(1) patch sampling
            rng = np.random.default_rng(abs(hash(visit.visit_id)) % (2**32))
            idx = rng.choice(len(fg_coords), size=20000, replace=False)
            fg_coords = fg_coords[idx]

        item: Dict[str, np.ndarray] = {
            "image": norm_vol,
            "fg_coords": fg_coords.astype(np.int64),
        }
        if target_mask is not None:
            item["mask"] = target_mask[None, ...]  # [1, H, W, D]
            lesion_coords = np.argwhere(target_mask > 0.5)
            item["lesion_coords"] = lesion_coords.astype(np.int64)
        if mask1 is not None:
            item["mask1"] = (mask1 > 0.5).astype(np.float32)[None, ...]
        if mask2 is not None:
            item["mask2"] = (mask2 > 0.5).astype(np.float32)[None, ...]

        if self.enabled:
            self._cache[visit.visit_id] = item
        return item

    def preload_visits(self, visits: Sequence[VisitRecord]) -> None:
        """Eagerly load a sequence of unique visits into RAM."""
        unique = {v.visit_id: v for v in visits}
        for v in unique.values():
            self.get_visit_data(v)


def extract_3d_patch_with_pad(
    volume: np.ndarray,
    center: Tuple[int, int, int],
    patch_size: Tuple[int, int, int] = (64, 64, 64),
) -> np.ndarray:
    """Extract a 3D patch of shape [C, pH, pW, pD] centered at `center` with zero-padding at borders."""
    _, h, w, d = volume.shape
    ph, pw, pd = patch_size
    cx, cy, cz = center

    x0, x1 = cx - ph // 2, cx - ph // 2 + ph
    y0, y1 = cy - pw // 2, cy - pw // 2 + pw
    z0, z1 = cz - pd // 2, cz - pd // 2 + pd

    # Clamp valid slice boundaries
    sx0, sx1 = max(0, x0), min(h, x1)
    sy0, sy1 = max(0, y0), min(w, y1)
    sz0, sz1 = max(0, z0), min(d, z1)

    patch = np.zeros((volume.shape[0], ph, pw, pd), dtype=volume.dtype)
    dx0, dy0, dz0 = sx0 - x0, sy0 - y0, sz0 - z0
    dx1, dy1, dz1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0), dz0 + (sz1 - sz0)

    if sx1 > sx0 and sy1 > sy0 and sz1 > sz0:
        patch[:, dx0:dx1, dy0:dy1, dz0:dz1] = volume[:, sx0:sx1, sy0:sy1, sz0:sz1]
    return patch


def apply_intensity_augmentation(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild random intensity scale, shift, and Gaussian noise to simulate inter-visit scanner drift."""
    aug = patch.copy()
    fg = np.any(np.abs(aug) > 1e-6, axis=0)
    if not np.any(fg):
        return aug

    for c in range(aug.shape[0]):
        scale = float(rng.uniform(0.92, 1.08))
        shift = float(rng.uniform(-0.08, 0.08))
        noise = rng.normal(0.0, 0.03, size=aug[c].shape).astype(np.float32)
        ch = aug[c] * scale + shift + noise
        aug[c] = np.where(fg, ch, 0.0)
    return aug


class AlignedLongitudinalSimCLRDataset(Dataset):
    """Samples anatomically aligned 3D foreground patches (p_t1, p_t2) from longitudinal visit pairs."""

    def __init__(
        self,
        pairs: Sequence[LongitudinalPairRecord],
        volume_cache: InMemoryVolumeCache,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        patches_per_pair: int = 16,
        max_spatial_jitter: int = 2,
        augment: bool = True,
    ) -> None:
        self.pairs = list(pairs)
        self.cache = volume_cache
        self.patch_size = tuple(patch_size)
        self.patches_per_pair = int(patches_per_pair)
        self.max_spatial_jitter = int(max_spatial_jitter)
        self.augment = augment

        all_Patients = sorted({p.patient_id for p in self.pairs})
        self.patient_to_int = {pid: idx for idx, pid in enumerate(all_Patients)}

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_pair

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pair_idx = index // self.patches_per_pair
        pair = self.pairs[pair_idx]

        data_t1 = self.cache.get_visit_data(pair.visit_t1)
        data_t2 = self.cache.get_visit_data(pair.visit_t2)

        rng = np.random.default_rng()
        fg_coords = data_t1["fg_coords"]
        center_t1 = tuple(int(v) for v in fg_coords[int(rng.integers(0, len(fg_coords)))])

        if self.augment and self.max_spatial_jitter > 0:
            jitter = rng.integers(-self.max_spatial_jitter, self.max_spatial_jitter + 1, size=3)
            center_t2 = (
                center_t1[0] + int(jitter[0]),
                center_t1[1] + int(jitter[1]),
                center_t1[2] + int(jitter[2]),
            )
        else:
            center_t2 = center_t1

        p_t1 = extract_3d_patch_with_pad(data_t1["image"], center_t1, self.patch_size)
        p_t2 = extract_3d_patch_with_pad(data_t2["image"], center_t2, self.patch_size)

        if self.augment:
            p_t1 = apply_intensity_augmentation(p_t1, rng)
            p_t2 = apply_intensity_augmentation(p_t2, rng)
            # Identical random 3D flips applied synchronously to both timepoints
            for axis in (1, 2, 3):
                if rng.random() < 0.5:
                    p_t1 = np.flip(p_t1, axis=axis).copy()
                    p_t2 = np.flip(p_t2, axis=axis).copy()

        return {
            "patch_t1": torch.from_numpy(p_t1),
            "patch_t2": torch.from_numpy(p_t2),
            "delta_t": torch.tensor([pair.delta_t_norm], dtype=torch.float32),
            "patient_label": torch.tensor(self.patient_to_int[pair.patient_id], dtype=torch.long),
            "center_coord": torch.tensor(center_t1, dtype=torch.float32),
        }


class LongitudinalSegmentationPatchDataset(Dataset):
    """Samples lesion- and evolution-enriched 3D patch pairs for supervised t2 mask prediction."""

    def __init__(
        self,
        pairs: Sequence[LongitudinalPairRecord],
        volume_cache: InMemoryVolumeCache,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        patches_per_pair: int = 16,
        pos_neg_ratio: float = 3.0,
        augment: bool = True,
    ) -> None:
        self.pairs = list(pairs)
        self.cache = volume_cache
        self.patch_size = tuple(patch_size)
        self.patches_per_pair = int(patches_per_pair)
        self.pos_prob = float(pos_neg_ratio) / (float(pos_neg_ratio) + 1.0)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_pair

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pair_idx = index // self.patches_per_pair
        pair = self.pairs[pair_idx]

        data_t1 = self.cache.get_visit_data(pair.visit_t1)
        data_t2 = self.cache.get_visit_data(pair.visit_t2)

        rng = np.random.default_rng()
        lesion_coords = data_t2.get("lesion_coords")
        fg_coords = data_t2["fg_coords"]

        if lesion_coords is not None and len(lesion_coords) > 0 and rng.random() < self.pos_prob:
            center = tuple(int(v) for v in lesion_coords[int(rng.integers(0, len(lesion_coords)))])
        else:
            center = tuple(int(v) for v in fg_coords[int(rng.integers(0, len(fg_coords)))])

        p_t1 = extract_3d_patch_with_pad(data_t1["image"], center, self.patch_size)
        p_t2 = extract_3d_patch_with_pad(data_t2["image"], center, self.patch_size)
        m_t1 = extract_3d_patch_with_pad(data_t1["mask"], center, self.patch_size)
        m_t2 = extract_3d_patch_with_pad(data_t2["mask"], center, self.patch_size)

        if self.augment:
            p_t1 = apply_intensity_augmentation(p_t1, rng)
            p_t2 = apply_intensity_augmentation(p_t2, rng)
            for axis in (1, 2, 3):
                if rng.random() < 0.5:
                    p_t1 = np.flip(p_t1, axis=axis).copy()
                    p_t2 = np.flip(p_t2, axis=axis).copy()
                    m_t1 = np.flip(m_t1, axis=axis).copy()
                    m_t2 = np.flip(m_t2, axis=axis).copy()

        return {
            "patch_t1": torch.from_numpy(p_t1),
            "patch_t2": torch.from_numpy(p_t2),
            "mask_t1": torch.from_numpy(m_t1),
            "mask_t2": torch.from_numpy(m_t2),
            "delta_t": torch.tensor([pair.delta_t_norm], dtype=torch.float32),
        }
