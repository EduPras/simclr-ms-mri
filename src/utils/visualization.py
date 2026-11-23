"""Portfolio-ready visualization utilities for longitudinal MS MRI and lesion evolution."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.engine.evaluator import compute_evolution_breakdown


# Color map for 3-state longitudinal lesion evolution overlay:
# 1 = DodgerBlue (Stable lesion: M_t1 & M_t2)
# 2 = Crimson Red (New / Enlarging lesion: M_t2 \ M_t1)
# 3 = Lime Green (Shrinking / Resolved lesion: M_t1 \ M_t2)
EVOLUTION_CMAP = ListedColormap(["#1E90FF", "#FF2E4C", "#00E676"])


def find_peak_lesion_slice(mask_3d: np.ndarray, axis: int = 2) -> int:
    """Return the slice index along `axis` with the maximum lesion area."""
    m = mask_3d[0] if mask_3d.ndim == 4 else mask_3d
    other_axes = tuple(i for i in range(3) if i != axis)
    counts = (m > 0.5).sum(axis=other_axes)
    if counts.max() == 0:
        return m.shape[axis] // 2
    return int(np.argmax(counts))


def plot_four_modalities_and_raters(
    image_4ch: np.ndarray,
    mask1: Optional[np.ndarray] = None,
    mask2: Optional[np.ndarray] = None,
    modality_names: Sequence[str] = ("FLAIR", "MPRAGE (T1)", "T2-w", "PD-w"),
    slice_idx: Optional[int] = None,
    title: str = "ISBI 2015 Co-Registered Multi-Sequence MRI & Expert Rater Masks",
) -> plt.Figure:
    """Plot axial slice across all 4 MRI sequences plus Rater 1 and Rater 2 lesion masks."""
    if slice_idx is None:
        ref_mask = mask1 if mask1 is not None else mask2
        slice_idx = find_peak_lesion_slice(ref_mask, axis=2) if ref_mask is not None else image_4ch.shape[-1] // 2

    n_cols = 4 + (1 if (mask1 is not None or mask2 is not None) else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 4.0), facecolor="#0E1117")
    fig.suptitle(f"{title} (Axial z={slice_idx})", color="white", fontsize=13, fontweight="bold")

    for c in range(4):
        ax = axes[c]
        sl = np.rot90(image_4ch[c, :, :, slice_idx])
        ax.imshow(sl, cmap="gray")
        ax.set_title(modality_names[c], color="white", fontsize=11)
        ax.axis("off")

    if n_cols == 5:
        ax = axes[4]
        flair_sl = np.rot90(image_4ch[0, :, :, slice_idx])
        ax.imshow(flair_sl, cmap="gray")
        if mask1 is not None:
            m1_sl = np.rot90((mask1[0] if mask1.ndim == 4 else mask1)[:, :, slice_idx])
            ax.contour(m1_sl, levels=[0.5], colors=["#FF3B30"], linewidths=1.5)
        if mask2 is not None:
            m2_sl = np.rot90((mask2[0] if mask2.ndim == 4 else mask2)[:, :, slice_idx])
            ax.contour(m2_sl, levels=[0.5], colors=["#00E676"], linewidths=1.5, linestyles="dashed")
        ax.set_title("Rater 1 (Red) vs Rater 2 (Green)", color="white", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_lesion_evolution_comparison(
    flair_t1: np.ndarray,
    flair_t2: np.ndarray,
    gt_mask_t1: np.ndarray,
    gt_mask_t2: np.ndarray,
    pred_phase_a_t2: np.ndarray,
    pred_phase_b_t2: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    patient_pair_title: str = "Longitudinal Lesion Evolution Comparison",
) -> plt.Figure:
    """Render portfolio figure comparing t1/t2 FLAIR, GT Evolution, Phase A, and Phase B SimCLR.

    Color legend:
        - Blue: Stable lesion (present at t1 & t2)
        - Red: New / Enlarging lesion (progression at t2)
        - Green: Shrinking / Resolved lesion (present at t1, resolved at t2)
    """
    m1 = gt_mask_t1[0] if gt_mask_t1.ndim == 4 else gt_mask_t1
    m2 = gt_mask_t2[0] if gt_mask_t2.ndim == 4 else gt_mask_t2
    change_vol = np.abs(m2 - m1)
    if slice_idx is None:
        slice_idx = find_peak_lesion_slice(change_vol if change_vol.max() > 0 else m2, axis=2)

    gt_evol = compute_evolution_breakdown(m1, m2)["state_map"]
    pa_evol = compute_evolution_breakdown(m1, pred_phase_a_t2)["state_map"]

    panels = [
        ("Visit t1 FLAIR + Mask t1", flair_t1, m1, False),
        ("Visit t2 FLAIR (Target Scan)", flair_t2, None, False),
        ("Ground Truth Evolution (t1→t2)", flair_t2, gt_evol, True),
        ("Phase A (Frozen MedicalNet)", flair_t2, pa_evol, True),
    ]
    if pred_phase_b_t2 is not None:
        pb_evol = compute_evolution_breakdown(m1, pred_phase_b_t2)["state_map"]
        panels.append(("Phase B (Longitudinal SimCLR)", flair_t2, pb_evol, True))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 4.2), facecolor="#0E1117")
    fig.suptitle(
        f"{patient_pair_title} | Axial Slice z={slice_idx}\n"
        "[Blue = Stable Plaque | Red = New/Enlarging Plaque | Green = Shrinking Plaque]",
        color="white",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (title, bg_vol, overlay, is_evol) in zip(axes, panels):
        bg_sl = np.rot90(bg_vol[:, :, slice_idx])
        ax.imshow(bg_sl, cmap="gray")
        if overlay is not None:
            ov_sl = np.rot90(overlay[:, :, slice_idx])
            if is_evol:
                masked_ov = np.ma.masked_where(ov_sl == 0, ov_sl)
                ax.imshow(masked_ov, cmap=EVOLUTION_CMAP, vmin=1, vmax=3, alpha=0.82, interpolation="nearest")
            else:
                masked_ov = np.ma.masked_where(ov_sl < 0.5, ov_sl)
                ax.imshow(masked_ov, cmap="cool", alpha=0.7, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10.5)
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_phase_comparison_bars(
    phase_a_summary: Dict[str, float],
    phase_b_summary: Dict[str, float],
    inter_rater_dice: Optional[float] = None,
) -> plt.Figure:
    """Create a portfolio summary bar chart comparing Phase A (Frozen) vs Phase B (Longitudinal SimCLR)."""
    metrics = ["t2_dice", "new_enlarging_dice", "stable_dice"]
    labels = ["Overall t2 Lesion Dice", "New/Enlarging Evolution Dice", "Stable Plaque Dice"]

    vals_a = [phase_a_summary.get(m, 0.0) for m in metrics]
    vals_b = [phase_b_summary.get(m, 0.0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    rects1 = ax.bar(x - width / 2, vals_a, width, label="Phase A: Frozen 3D MedicalNet", color="#5C7CFA")
    rects2 = ax.bar(x + width / 2, vals_b, width, label="Phase B: Longitudinal SimCLR + Δt Adapted", color="#FF5252")

    if inter_rater_dice is not None:
        ax.axhline(
            inter_rater_dice,
            color="#20C997",
            linestyle="--",
            linewidth=2.0,
            label=f"Human Inter-Rater Ceiling (Rater 1 vs 2 = {inter_rater_dice:.3f})",
        )

    ax.set_ylabel("Dice Similarity Coefficient (↑)", fontsize=11)
    ax.set_title("Downstream Longitudinal MS Lesion Evolution: Phase A vs. Phase B SimCLR", fontsize=12.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig
