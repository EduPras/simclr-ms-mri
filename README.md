# Longitudinal MS Brain MRI Analysis with Pretrained 3D MedicalNet & Contrastive Learning

A modular PyTorch + MONAI deep learning pipeline for **longitudinal Multiple Sclerosis (MS) white matter lesion evolution analysis** on the **ISBI 2015 Longitudinal MS Lesion Dataset**.

Given a pair of co-registered longitudinal brain MRI scans $(x_{t_1}, x_{t_2})$ across 4 sequences (`FLAIR`, `MPRAGE/T1-w`, `T2-w`, `PD-w`) separated by visit interval $\Delta t = t_2 - t_1$, this pipeline predicts the **evolved $t_2$ 3D lesion segmentation mask ($\hat{M}_{t_2}$)** and derives a 3-class **Longitudinal Lesion Evolution Map**:
- **Stable Lesion:** $M_{t_1} \cap \hat{M}_{t_2}$
- **New / Enlarging Lesion (Disease Progression):** $\hat{M}_{t_2} \setminus M_{t_1}$
- **Shrinking / Resolving Lesion:** $M_{t_1} \setminus \hat{M}_{t_2}$

---

## Key Architectural Highlights

1. **3D Volumetric Patch Backbone with 4-Channel Weight Inflation (`src/models/backbone.py`)**
   - Uses MONAI's pretrained **MedicalNet 3D ResNet-18** (`resnet18`, Med3D pretrained on 23 3D medical datasets).
   - Adapts the 1-channel pretrained 3D stem (`conv1` shape `[64, 1, 7, 7, 7]`) to 4 MRI sequences (`[FLAIR, MPRAGE, T2, PD]`) via **Weight Inflation** ($W_{4\text{ch}} = \frac{1}{4}\text{repeat}(W_{1\text{ch}}, 4)$), preserving pretrained spatial filter statistics at initialization.

2. **Two-Phase Longitudinal Representation Learning**
   - **Phase A (Frozen Pretrained Baseline):** Freezes 100% of the 3D MedicalNet encoder and trains the **Siamese Multi-Scale Temporal-Difference 3D U-Net Decoder** (`SiameseLongitudinalUNet3D`) using $[F^{(l)}_{t_2},\; F^{(l)}_{t_2} - F^{(l)}_{t_1},\; |F^{(l)}_{t_2} - F^{(l)}_{t_1}|]$ and **FiLM ($\Delta t$) time-interval conditioning**.
   - **Phase B (Hybrid Longitudinal SimCLR + Auxiliary $\Delta t$ Head):** Fine-tunes `conv1`, `bn1`, and `layer4` on anatomically aligned $64^3$ foreground brain patch pairs across visits using **Patient-Masked NT-Xent Contrastive Loss** + an **Auxiliary Temporal Interval ($\Delta t$) Head** $h_\psi(z_{t_2} - z_{t_1}) \rightarrow \widehat{\Delta t}$ that prevents longitudinal lesion differences from collapsing to zero.

3. **Strict Zero-Leakage Patient-Level Validation (`src/data/isbi_dataset.py`)**
   - Implements **5-Fold Leave-One-Patient-Out Cross-Validation (LOPO-CV)** across the 5 labeled ISBI 2015 subjects (`training01`–`training05`).
   - In every fold, the held-out validation patient is excluded from **both** Longitudinal SimCLR pretraining and supervised segmentation training.
   - Leverages the 14 unlabeled ISBI 2015 test subjects (`test01`–`test14`, 61 scans) + 4 active training patients (18 patients total, 136 forward visit pairs) for self-supervised Phase B SimCLR adaptation.

4. **Google Colab Pro (A100 GPU) Optimizations**
   - Whole-dataset in-RAM caching (`monai.data.CacheDataset(cache_rate=1.0)`) to eliminate Google Drive I/O bottlenecks.
   - Native `bfloat16` mixed-precision training (`torch.amp.autocast`).
   - Synchronous paired full-volume ($181 \times 217 \times 181$) 3D `sliding_window_inference` with Gaussian overlap blending.

---

## Project Structure

```text
simclr-ms-mri/
├── configs/
│   └── default.yaml                 # Centralized hyperparameters for Data, SimCLR, Decoder, and A100 runtime
├── src/
│   ├── data/
│   │   ├── isbi_dataset.py          # Flexible ISBI 2015 NIfTI scanner, visit-pair builder, zero-leakage LOPO splits
│   │   └── transforms.py            # Foreground Z-score normalization, aligned SimCLR & lesion-aware 3D patch samplers
│   ├── models/
│   │   ├── backbone.py              # 3D MedicalNet ResNet-18 with 4-channel weight inflation & phase freeze control
│   │   ├── simclr.py                # LongitudinalSimCLR3D (MLP projection head + auxiliary Δt difference head)
│   │   └── segmentation_decoder.py  # SiameseLongitudinalUNet3D with multi-scale differencing & FiLM(Δt) modulation
│   ├── losses/
│   │   ├── contrastive_loss.py      # PatientMaskedNTXentLoss & HybridLongitudinalSSLLoss
│   │   └── segmentation_loss.py     # EvolutionWeightedDiceFocalLoss emphasizing dynamic lesion voxels
│   ├── engine/
│   │   ├── trainer.py               # BF16/FP16 AMP training loops for Phase B SimCLR and Downstream Segmentation
│   │   └── evaluator.py             # Paired 3D sliding-window full-volume inference & longitudinal evolution metrics
│   └── utils/
│       ├── config.py                # YAML configuration loader & deterministic seeding
│       └── visualization.py         # Portfolio figures: 3D evolution maps, feature difference heatmaps, trajectories
├── notebooks/
│   ├── 01_data_exploration.ipynb           # EDA, 4-modality orthoslices, lesion volume trajectories, inter-rater Dice
│   ├── 02_phase_a_frozen_baseline.ipynb    # Phase A: Frozen MedicalNet + Siamese Temporal-Difference Decoder
│   └── 03_phase_b_simclr_and_eval.ipynb    # Phase B: Hybrid Longitudinal SimCLR + Phase A vs. Phase B benchmark
└── tests/
    └── test_pipeline.py             # Automated unit & integration tests on synthetic 3D volumes
```

---

## Quickstart (Local & Google Colab Pro)

### 1. Local Installation & Unit Testing
```bash
pip install -r requirements.txt
pip install -e .
pytest tests/test_pipeline.py -v
```

### 2. Running in Google Colab Pro (A100 GPU)
1. Place the ISBI 2015 dataset under `/content/drive/MyDrive/masters/data` on Google Drive.
2. Upload or clone this repository to `/content/drive/MyDrive/masters/simclr-ms-mri` (or clone directly from GitHub inside the notebook).
3. Open and run the milestone notebooks in order:
   - `notebooks/01_data_exploration.ipynb`
   - `notebooks/02_phase_a_frozen_baseline.ipynb`
   - `notebooks/03_phase_b_simclr_and_eval.ipynb`
