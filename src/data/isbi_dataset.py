"""ISBI 2015 Longitudinal MS Lesion Dataset scanner, visit-pair builder, and zero-leakage LOPO splits."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


DEFAULT_MODALITIES: Tuple[str, ...] = ("flair", "mprage", "t2", "pd")

# Regex matching standard ISBI 2015 preprocessed NIfTI filenames, e.g.:
# training01_01_flair_pp.nii, training01_01_mask1.nii, test02_03_mprage_pp.nii.gz
ISBI_FILENAME_PATTERN = re.compile(
    r"^(?P<subject>(?:training|test)\d+)_(?P<visit>\d+)_(?P<tag>[a-zA-Z0-9]+)(?:_pp)?\.nii(?:\.gz)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VisitRecord:
    """Metadata and file paths for a single patient timepoint (visit)."""

    patient_id: str
    visit_idx: int
    is_labeled: bool
    modality_paths: Dict[str, Path]
    mask1_path: Optional[Path] = None
    mask2_path: Optional[Path] = None

    @property
    def visit_id(self) -> str:
        return f"{self.patient_id}_{self.visit_idx:02d}"


@dataclass(frozen=True)
class LongitudinalPairRecord:
    """An ordered longitudinal pair of visits (t1 < t2) from the same patient."""

    patient_id: str
    visit_t1: VisitRecord
    visit_t2: VisitRecord
    delta_t_steps: int
    delta_t_norm: float
    is_labeled: bool

    @property
    def pair_id(self) -> str:
        return f"{self.patient_id}_t{self.visit_t1.visit_idx:02d}_to_t{self.visit_t2.visit_idx:02d}"


@dataclass
class LOPOFoldSplit:
    """Strict zero-leakage patient-level split for a Leave-One-Patient-Out CV fold."""

    fold_idx: int
    val_patient_id: str
    train_labeled_patient_ids: List[str]
    unlabeled_patient_ids: List[str]
    simclr_patient_ids: List[str]
    simclr_pairs: List[LongitudinalPairRecord] = field(default_factory=list)
    seg_train_pairs: List[LongitudinalPairRecord] = field(default_factory=list)
    seg_val_pairs: List[LongitudinalPairRecord] = field(default_factory=list)

    def verify_zero_leakage(self) -> None:
        """Raise AssertionError if the held-out validation patient leaks into any training pool."""
        simclr_pids = {p.patient_id for p in self.simclr_pairs}
        seg_train_pids = {p.patient_id for p in self.seg_train_pairs}
        seg_val_pids = {p.patient_id for p in self.seg_val_pairs}

        assert self.val_patient_id not in simclr_pids, (
            f"DATA LEAKAGE DETECTED: Validation patient {self.val_patient_id} found in SimCLR pool!"
        )
        assert self.val_patient_id not in seg_train_pids, (
            f"DATA LEAKAGE DETECTED: Validation patient {self.val_patient_id} found in Seg Train pool!"
        )
        assert simclr_pids.isdisjoint(seg_val_pids), (
            f"DATA LEAKAGE DETECTED: Overlap between SimCLR and Val patients: {simclr_pids & seg_val_pids}"
        )
        assert seg_train_pids.isdisjoint(seg_val_pids), (
            f"DATA LEAKAGE DETECTED: Overlap between Seg Train and Val patients: {seg_train_pids & seg_val_pids}"
        )


def discover_isbi2015_visits(
    root_dir: Union[str, Path],
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> Dict[str, List[VisitRecord]]:
    """Recursively scan root_dir for ISBI 2015 NIfTI files and organize chronologically by patient.

    Supports both flat folder layouts and nested subject folders (e.g., training01/preprocessed/*.nii).
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"ISBI 2015 data directory does not exist: {root}")

    req_mods = [m.lower() for m in modalities]
    grouped_files: Dict[Tuple[str, int], Dict[str, Path]] = {}

    for path in sorted(root.rglob("*.nii*")):
        if not path.is_file():
            continue
        match = ISBI_FILENAME_PATTERN.match(path.name)
        if not match:
            continue

        subject = match.group("subject").lower()
        visit_idx = int(match.group("visit"))
        tag = match.group("tag").lower()

        # Normalize common tag aliases (e.g., t1 -> mprage if needed)
        if tag == "t1" and "mprage" in req_mods:
            tag = "mprage"

        key = (subject, visit_idx)
        if key not in grouped_files:
            grouped_files[key] = {}
        grouped_files[key][tag] = path

    patient_visits: Dict[str, List[VisitRecord]] = {}
    for (subject, visit_idx), tag_map in sorted(grouped_files.items()):
        if not all(m in tag_map for m in req_mods):
            continue

        mod_paths = {m: tag_map[m] for m in req_mods}
        mask1 = tag_map.get("mask1")
        mask2 = tag_map.get("mask2")
        is_labeled = mask1 is not None or mask2 is not None

        record = VisitRecord(
            patient_id=subject,
            visit_idx=visit_idx,
            is_labeled=is_labeled,
            modality_paths=mod_paths,
            mask1_path=mask1,
            mask2_path=mask2,
        )
        patient_visits.setdefault(subject, []).append(record)

    # Sort each patient's visits chronologically
    for pid in patient_visits:
        patient_visits[pid].sort(key=lambda v: v.visit_idx)

    if not patient_visits:
        raise RuntimeError(
            f"No valid ISBI 2015 visits with modalities {req_mods} found under {root}."
        )

    return patient_visits


def build_longitudinal_pairs(
    patient_visits: Dict[str, List[VisitRecord]],
    max_time_gap_norm: float = 4.0,
    consecutive_only: bool = False,
) -> Dict[str, List[LongitudinalPairRecord]]:
    """Construct all forward longitudinal visit pairs (t_i, t_j) with t_i < t_j per patient."""
    patient_pairs: Dict[str, List[LongitudinalPairRecord]] = {}

    for pid, visits in sorted(patient_visits.items()):
        pairs: List[LongitudinalPairRecord] = []
        n_visits = len(visits)
        for i in range(n_visits):
            for j in range(i + 1, n_visits):
                if consecutive_only and (j - i) > 1:
                    continue
                v1, v2 = visits[i], visits[j]
                dt_steps = v2.visit_idx - v1.visit_idx
                dt_norm = float(dt_steps) / float(max_time_gap_norm)
                pairs.append(
                    LongitudinalPairRecord(
                        patient_id=pid,
                        visit_t1=v1,
                        visit_t2=v2,
                        delta_t_steps=dt_steps,
                        delta_t_norm=dt_norm,
                        is_labeled=(v1.is_labeled and v2.is_labeled),
                    )
                )
        if pairs:
            patient_pairs[pid] = pairs

    return patient_pairs


def get_lopo_fold_splits(
    patient_visits: Dict[str, List[VisitRecord]],
    fold_idx: int = 0,
    max_time_gap_norm: float = 4.0,
    consecutive_only: bool = False,
) -> LOPOFoldSplit:
    """Create a strict zero-leakage Leave-One-Patient-Out (LOPO) fold split.

    - Labeled patients (e.g., training01..training05) are split into:
      * 1 validation patient (held out from BOTH SimCLR and supervised segmentation)
      * N-1 active labeled training patients
    - Unlabeled patients (e.g., test01..test14) are combined with the N-1 active training
      patients to form the Phase B Longitudinal SimCLR pool.
    """
    patient_pairs = build_longitudinal_pairs(
        patient_visits,
        max_time_gap_norm=max_time_gap_norm,
        consecutive_only=consecutive_only,
    )

    labeled_patients = sorted(
        [pid for pid, visits in patient_visits.items() if all(v.is_labeled for v in visits)]
    )
    unlabeled_patients = sorted(
        [pid for pid in patient_visits.keys() if pid not in labeled_patients]
    )

    if len(labeled_patients) < 2:
        raise ValueError(
            f"Need at least 2 labeled patients for LOPO cross-validation, found {len(labeled_patients)}."
        )
    if not (0 <= fold_idx < len(labeled_patients)):
        raise ValueError(
            f"fold_idx must be in [0, {len(labeled_patients) - 1}], got {fold_idx}."
        )

    val_pid = labeled_patients[fold_idx]
    train_labeled_pids = [pid for pid in labeled_patients if pid != val_pid]
    simclr_pids = sorted(train_labeled_pids + unlabeled_patients)

    simclr_pairs = [pair for pid in simclr_pids for pair in patient_pairs.get(pid, [])]
    seg_train_pairs = [pair for pid in train_labeled_pids for pair in patient_pairs.get(pid, [])]
    seg_val_pairs = list(patient_pairs.get(val_pid, []))

    split = LOPOFoldSplit(
        fold_idx=fold_idx,
        val_patient_id=val_pid,
        train_labeled_patient_ids=train_labeled_pids,
        unlabeled_patient_ids=unlabeled_patients,
        simclr_patient_ids=simclr_pids,
        simclr_pairs=simclr_pairs,
        seg_train_pairs=seg_train_pairs,
        seg_val_pairs=seg_val_pairs,
    )
    split.verify_zero_leakage()
    return split
