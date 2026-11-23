"""End-to-end automated verification suite for the Longitudinal MS MRI SimCLR & Segmentation pipeline."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.isbi_dataset import (
    build_longitudinal_pairs,
    discover_isbi2015_visits,
    get_lopo_fold_splits,
)
from src.data.transforms import (
    AlignedLongitudinalSimCLRDataset,
    InMemoryVolumeCache,
    LongitudinalSegmentationPatchDataset,
)
from src.engine.evaluator import (
    compute_evolution_breakdown,
    compute_longitudinal_pair_metrics,
    predict_full_volume_pair,
)
from src.engine.trainer import train_segmentation_epoch, train_simclr_epoch
from src.losses.contrastive_loss import HybridLongitudinalSSLLoss, PatientMaskedNTXentLoss
from src.losses.segmentation_loss import EvolutionWeightedDiceFocalLoss
from src.models.backbone import MedicalNet3DResNet18, inflate_conv3d_from_1ch
from src.models.segmentation_decoder import SiameseLongitudinalUNet3D
from src.models.simclr import LongitudinalSimCLR3D
from src.utils.config import load_config, set_deterministic_seed


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def synthetic_isbi_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a synthetic ISBI 2015 directory with 3 labeled and 2 unlabeled subjects (3D NIfTI)."""
    root = tmp_path_factory.mktemp("synthetic_isbi2015")
    rng = np.random.default_rng(42)
    shape = (48, 48, 48)
    affine = np.eye(4, dtype=np.float32)

    subjects = [
        ("training01", 3, True),
        ("training02", 3, True),
        ("training03", 3, True),
        ("test01", 3, False),
        ("test02", 3, False),
    ]

    for pid, n_visits, is_labeled in subjects:
        subj_dir = root / pid / "preprocessed"
        subj_dir.mkdir(parents=True, exist_ok=True)
        for v_idx in range(1, n_visits + 1):
            for mod in ("flair", "mprage", "t2", "pd"):
                vol = np.zeros(shape, dtype=np.float32)
                # Brain foreground sphere/box
                vol[8:40, 8:40, 8:40] = rng.normal(100.0, 15.0, size=(32, 32, 32)).astype(np.float32)
                nib.save(
                    nib.Nifti1Image(vol, affine),
                    str(subj_dir / f"{pid}_{v_idx:02d}_{mod}_pp.nii.gz"),
                )
            if is_labeled:
                m1 = np.zeros(shape, dtype=np.float32)
                m2 = np.zeros(shape, dtype=np.float32)
                # Growing lesion across timepoints v_idx
                radius = 3 + v_idx
                m1[20 : 20 + radius, 20 : 20 + radius, 20 : 20 + radius] = 1.0
                m2[20 : 19 + radius, 20 : 20 + radius, 20 : 20 + radius] = 1.0
                nib.save(nib.Nifti1Image(m1, affine), str(subj_dir / f"{pid}_{v_idx:02d}_mask1.nii.gz"))
                nib.save(nib.Nifti1Image(m2, affine), str(subj_dir / f"{pid}_{v_idx:02d}_mask2.nii.gz"))

    return root


def test_config_and_zero_leakage_lopo_splits(synthetic_isbi_dir: Path) -> None:
    """Verify ISBI 2015 NIfTI discovery and strict patient-level zero-leakage LOPO splits."""
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    assert cfg["backbone"]["in_channels"] == 4

    patient_visits = discover_isbi2015_visits(synthetic_isbi_dir)
    assert len(patient_visits) == 5

    pairs = build_longitudinal_pairs(patient_visits)
    # Each subject has 3 visits -> 3 choose 2 = 3 forward pairs
    assert sum(len(p) for p in pairs.values()) == 15

    for fold_idx in range(3):
        split = get_lopo_fold_splits(patient_visits, fold_idx=fold_idx)
        split.verify_zero_leakage()
        assert len(split.seg_val_pairs) == 3
        assert len(split.seg_train_pairs) == 6
        assert len(split.simclr_pairs) == 12  # 2 train labeled + 2 unlabeled = 4 subjects * 3 pairs


def test_4channel_conv3d_weight_inflation() -> None:
    """Verify that 1-to-4 channel weight inflation preserves exact activation outputs on repeated inputs."""
    set_deterministic_seed(123)
    conv_1ch = torch.nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    conv_4ch = inflate_conv3d_from_1ch(conv_1ch, in_channels=4)

    x_1ch = torch.randn(2, 1, 16, 16, 16)
    x_4ch = x_1ch.repeat(1, 4, 1, 1, 1)

    out_1ch = conv_1ch(x_1ch)
    out_4ch = conv_4ch(x_4ch)
    assert torch.allclose(out_1ch, out_4ch, atol=1e-5)


def test_patient_masked_nt_xent_and_hybrid_ssl_loss() -> None:
    """Verify PatientMaskedNTXentLoss masks false negatives from the same patient without NaNs."""
    set_deterministic_seed(42)
    loss_fn = HybridLongitudinalSSLLoss(temperature=0.1, lambda_delta_t=0.25)

    proj_t1 = torch.randn(6, 32, requires_grad=True)
    proj_t2 = proj_t1.detach().clone() + 0.05 * torch.randn(6, 32)
    proj_t2.requires_grad_(True)
    pred_dt = torch.tensor([[0.25], [0.5], [0.25], [0.75], [0.5], [0.25]], requires_grad=True)
    target_dt = torch.tensor([[0.25], [0.5], [0.25], [0.75], [0.5], [0.25]])
    # Duplicate patient labels in batch to exercise false-negative masking
    patient_labels = torch.tensor([0, 0, 1, 1, 2, 0], dtype=torch.long)

    out = loss_fn(proj_t1, proj_t2, pred_dt, target_dt, patient_labels=patient_labels)
    assert torch.isfinite(out["loss"])
    assert out["top1_acc"].item() >= 0.8
    out["loss"].backward()
    assert proj_t1.grad is not None and torch.all(torch.isfinite(proj_t1.grad))


def test_end_to_end_simclr_and_segmentation_pipeline(synthetic_isbi_dir: Path) -> None:
    """Run 1 step of Phase B SimCLR, 1 step of Siamese 3D U-Net training, and full-volume 3D inference."""
    set_deterministic_seed(42)
    device = torch.device("cpu")
    patch_size = (32, 32, 32)

    patient_visits = discover_isbi2015_visits(synthetic_isbi_dir)
    split = get_lopo_fold_splits(patient_visits, fold_idx=0)
    cache = InMemoryVolumeCache(enabled=True)

    # 1. Phase B SimCLR step
    simclr_ds = AlignedLongitudinalSimCLRDataset(
        pairs=split.simclr_pairs[:4],
        volume_cache=cache,
        patch_size=patch_size,
        patches_per_pair=2,
        augment=True,
    )
    simclr_loader = DataLoader(simclr_ds, batch_size=4, shuffle=True)

    backbone = MedicalNet3DResNet18(in_channels=4, pretrained=False)
    simclr_model = LongitudinalSimCLR3D(
        backbone=backbone,
        proj_hidden_dim=64,
        proj_out_dim=32,
        unfreeze_layers=("conv1", "bn1", "layer4"),
    ).to(device)

    # Verify layer1-3 are frozen while conv1 and layer4 are trainable
    assert not any(p.requires_grad for p in simclr_model.backbone.backbone.layer1.parameters())
    assert any(p.requires_grad for p in simclr_model.backbone.backbone.layer4.parameters())

    ssl_criterion = HybridLongitudinalSSLLoss(temperature=0.1, lambda_delta_t=0.25)
    ssl_opt = torch.optim.AdamW([p for p in simclr_model.parameters() if p.requires_grad], lr=1e-3)
    ssl_metrics = train_simclr_epoch(
        model=simclr_model,
        dataloader=simclr_loader,
        criterion=ssl_criterion,
        optimizer=ssl_opt,
        device=device,
        use_amp=False,
    )
    assert np.isfinite(ssl_metrics["loss"])

    # 2. Downstream Siamese Longitudinal Segmentation step
    seg_ds = LongitudinalSegmentationPatchDataset(
        pairs=split.seg_train_pairs[:2],
        volume_cache=cache,
        patch_size=patch_size,
        patches_per_pair=2,
        augment=True,
    )
    seg_loader = DataLoader(seg_ds, batch_size=2, shuffle=True)

    seg_model = SiameseLongitudinalUNet3D(
        backbone=backbone,
        decoder_channels=(64, 32, 16, 16),
        freeze_backbone=True,
    ).to(device)
    seg_criterion = EvolutionWeightedDiceFocalLoss(change_voxel_weight=2.0)
    seg_opt = torch.optim.AdamW([p for p in seg_model.parameters() if p.requires_grad], lr=1e-3)

    seg_metrics = train_segmentation_epoch(
        model=seg_model,
        dataloader=seg_loader,
        criterion=seg_criterion,
        optimizer=seg_opt,
        device=device,
        use_amp=False,
    )
    assert np.isfinite(seg_metrics["loss"])

    # 3. Full-volume 3D sliding-window inference & evolution metrics
    val_pair = split.seg_val_pairs[0]
    d1 = cache.get_visit_data(val_pair.visit_t1)
    d2 = cache.get_visit_data(val_pair.visit_t2)

    pred_out = predict_full_volume_pair(
        model=seg_model,
        vol_t1=d1["image"],
        vol_t2=d2["image"],
        delta_t_norm=val_pair.delta_t_norm,
        device=device,
        roi_size=patch_size,
        sw_batch_size=2,
        overlap=0.25,
        use_amp=False,
    )
    assert pred_out["pred_mask_t2"].shape == (48, 48, 48)

    evol = compute_evolution_breakdown(d1["mask"][0], d2["mask"][0])
    # Because synthetic lesion grew from visit 1 to visit 2, new_or_enlarging must be non-empty
    assert evol["new_or_enlarging"].sum() > 0
    assert evol["stable"].sum() > 0

    pair_metrics = compute_longitudinal_pair_metrics(
        pred_mask_t2=pred_out["pred_mask_t2"],
        gt_mask_t2=d2["mask"][0],
        gt_mask_t1=d1["mask"][0],
    )
    assert 0.0 <= pair_metrics["t2_dice"] <= 1.0


def test_notebooks_valid_json_and_python_syntax() -> None:
    """Verify all 3 milestone Colab notebooks are valid nbformat v4 JSON and pass Python AST syntax checks."""
    nb_dir = PROJECT_ROOT / "notebooks"
    notebooks = [
        nb_dir / "01_data_exploration.ipynb",
        nb_dir / "02_phase_a_frozen_baseline.ipynb",
        nb_dir / "03_phase_b_simclr_and_eval.ipynb",
    ]
    for nb_path in notebooks:
        assert nb_path.exists(), f"Missing notebook: {nb_path}"
        with nb_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)
        assert nb.get("nbformat") == 4
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        assert len(code_cells) >= 4
        for idx, cell in enumerate(code_cells):
            source_code = "".join(cell.get("source", []))
            ast.parse(source_code, filename=f"{nb_path.name}#cell_{idx}")
