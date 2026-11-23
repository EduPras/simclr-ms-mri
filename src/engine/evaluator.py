"""Full-volume 3D sliding-window inference and longitudinal MS lesion evolution metrics."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from src.data.isbi_dataset import LongitudinalPairRecord
from src.data.transforms import InMemoryVolumeCache
from src.models.segmentation_decoder import SiameseLongitudinalUNet3D
from src.utils.config import get_amp_dtype


def compute_binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-5) -> float:
    """Compute binary Dice similarity coefficient between 3D masks."""
    p = (pred > 0.5).astype(np.float32)
    t = (target > 0.5).astype(np.float32)
    p_sum = float(p.sum())
    t_sum = float(t.sum())
    if p_sum == 0.0 and t_sum == 0.0:
        return 1.0
    inter = float((p * t).sum())
    return float((2.0 * inter + eps) / (p_sum + t_sum + eps))


def compute_evolution_breakdown(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Decompose longitudinal lesion change between t1 and t2 into 3 clinical states.

    States:
        - stable: present at both t1 and t2 (M_t1 & M_t2)
        - new_or_enlarging: absent at t1, present at t2 (M_t2 & ~M_t1)
        - shrinking_or_resolved: present at t1, absent at t2 (M_t1 & ~M_t2)
        - state_map: integer label map (0=background, 1=stable, 2=new/enlarging, 3=shrinking)
    """
    m1 = (mask_t1 > 0.5)
    m2 = (mask_t2 > 0.5)
    if m1.ndim == 4:
        m1 = m1[0]
    if m2.ndim == 4:
        m2 = m2[0]

    stable = m1 & m2
    new_or_enlarging = m2 & (~m1)
    shrinking = m1 & (~m2)

    state_map = np.zeros(m2.shape, dtype=np.uint8)
    state_map[stable] = 1
    state_map[new_or_enlarging] = 2
    state_map[shrinking] = 3

    return {
        "stable": stable.astype(np.float32),
        "new_or_enlarging": new_or_enlarging.astype(np.float32),
        "shrinking": shrinking.astype(np.float32),
        "state_map": state_map,
    }


def compute_longitudinal_pair_metrics(
    pred_mask_t2: np.ndarray,
    gt_mask_t2: np.ndarray,
    gt_mask_t1: np.ndarray,
    voxel_volume_ml: float = 0.001,  # 1 mm^3 in MNI space = 0.001 mL
) -> Dict[str, float]:
    """Compute comprehensive clinical metrics for t2 lesion mask and longitudinal evolution."""
    pred_evol = compute_evolution_breakdown(gt_mask_t1, pred_mask_t2)
    gt_evol = compute_evolution_breakdown(gt_mask_t1, gt_mask_t2)

    overall_t2_dice = compute_binary_dice(pred_mask_t2, gt_mask_t2)
    new_lesion_dice = compute_binary_dice(pred_evol["new_or_enlarging"], gt_evol["new_or_enlarging"])
    stable_lesion_dice = compute_binary_dice(pred_evol["stable"], gt_evol["stable"])

    # Lesion load & longitudinal volume change (ΔV = V_t2 - V_t1) in mL
    v_t1_ml = float((gt_mask_t1 > 0.5).sum()) * voxel_volume_ml
    v_t2_gt_ml = float((gt_mask_t2 > 0.5).sum()) * voxel_volume_ml
    v_t2_pred_ml = float((pred_mask_t2 > 0.5).sum()) * voxel_volume_ml

    delta_v_gt_ml = v_t2_gt_ml - v_t1_ml
    delta_v_pred_ml = v_t2_pred_ml - v_t1_ml
    delta_v_abs_error_ml = abs(delta_v_pred_ml - delta_v_gt_ml)

    return {
        "t2_dice": overall_t2_dice,
        "new_enlarging_dice": new_lesion_dice,
        "stable_dice": stable_lesion_dice,
        "vol_t1_ml": v_t1_ml,
        "vol_t2_gt_ml": v_t2_gt_ml,
        "vol_t2_pred_ml": v_t2_pred_ml,
        "delta_vol_gt_ml": delta_v_gt_ml,
        "delta_vol_pred_ml": delta_v_pred_ml,
        "delta_vol_mae_ml": delta_v_abs_error_ml,
    }


@torch.no_grad()
def predict_full_volume_pair(
    model: SiameseLongitudinalUNet3D,
    vol_t1: np.ndarray,
    vol_t2: np.ndarray,
    delta_t_norm: float,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    sw_batch_size: int = 16,
    overlap: float = 0.5,
    threshold: float = 0.5,
    use_amp: bool = True,
    amp_dtype_str: str = "bfloat16",
) -> Dict[str, np.ndarray]:
    """Predict full 3D t2 lesion probability & binary mask via synchronous sliding-window inference."""
    model.eval()
    amp_enabled = use_amp and (device.type == "cuda")
    amp_dtype = get_amp_dtype(amp_dtype_str) if amp_enabled else torch.float32

    # Stack [vol_t1, vol_t2] along channel dimension -> [1, 8, H, W, D]
    stacked = np.concatenate([vol_t1, vol_t2], axis=0)[None, ...]
    x_tensor = torch.from_numpy(stacked).to(device)
    dt_tensor = torch.tensor([[delta_t_norm]], dtype=torch.float32, device=device)

    def _predictor(window_batch: torch.Tensor) -> torch.Tensor:
        return model.forward_stacked_pair(window_batch, dt_tensor)

    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = sliding_window_inference(
            inputs=x_tensor,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=_predictor,
            overlap=overlap,
            mode="gaussian",
        )
        probs = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

    pred_mask = (probs >= threshold).astype(np.float32)
    return {
        "prob_t2": probs,
        "pred_mask_t2": pred_mask,
    }


@torch.no_grad()
def evaluate_lopo_validation_pairs(
    model: SiameseLongitudinalUNet3D,
    val_pairs: Sequence[LongitudinalPairRecord],
    volume_cache: InMemoryVolumeCache,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    sw_batch_size: int = 16,
    overlap: float = 0.5,
    threshold: float = 0.5,
    use_amp: bool = True,
    amp_dtype_str: str = "bfloat16",
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Evaluate all longitudinal visit pairs of the held-out validation patient."""
    per_pair_results: List[Dict[str, float]] = []

    for pair in val_pairs:
        d1 = volume_cache.get_visit_data(pair.visit_t1)
        d2 = volume_cache.get_visit_data(pair.visit_t2)

        pred_out = predict_full_volume_pair(
            model=model,
            vol_t1=d1["image"],
            vol_t2=d2["image"],
            delta_t_norm=pair.delta_t_norm,
            device=device,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            threshold=threshold,
            use_amp=use_amp,
            amp_dtype_str=amp_dtype_str,
        )

        metrics = compute_longitudinal_pair_metrics(
            pred_mask_t2=pred_out["pred_mask_t2"],
            gt_mask_t2=d2["mask"][0],
            gt_mask_t1=d1["mask"][0],
        )
        metrics["delta_t_steps"] = float(pair.delta_t_steps)
        per_pair_results.append(metrics)

    if not per_pair_results:
        return {}, []

    keys = [k for k in per_pair_results[0].keys() if k != "delta_t_steps"]
    summary = {k: float(np.mean([r[k] for r in per_pair_results])) for k in keys}
    return summary, per_pair_results
