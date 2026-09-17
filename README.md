# Large-Scale Data Analytics and AI-Based Detection of Thyroid Nodules Using Ultrasound Images

**Subtitle:** Transforming large-scale thyroid ultrasound data into analytical insights and an AI-based detection system

An academic engineering project combining large-scale image data analytics, leakage-aware dataset preparation, statistical analysis, transfer learning, and a desktop AI classification application for thyroid ultrasound images.

> **Academic research / educational prototype only.** The application is not intended for clinical diagnosis, treatment, or replacement of qualified medical professionals.

## Project overview

The project has two connected components:

1. **Data Analytics** — extract measurable morphological, intensity, texture, and image-level features from thousands of annotated thyroid ultrasound images and study their relationship with benign/malignant labels.
2. **AI Classification** — train and evaluate EfficientNetB0-based binary classifiers and deploy the final nodule-focused V3 model in a desktop application.

## Pipeline

```text
TN5000 dataset
      │
      ▼
Dataset validation
      │
      ▼
Exact duplicate / split-leakage analysis
      │
      ▼
Leakage-safe modeling split
      │
      ├──────────────► Analytical feature dataset
      │                       │
      │                       ├── EDA
      │                       └── Statistical analysis
      │
      ▼
Nodule crop generation
      │
      ▼
EfficientNetB0 training
      │
      ├── V1: full image
      ├── V2: full image + refined training
      └── V3: nodule-focused crop
      │
      ▼
Final V3 model
      │
      ▼
Manual ROI selection in desktop application
      │
      ▼
10% padded ROI → 224×224 → EfficientNet preprocessing
      │
      ▼
Benign / Malignant probability
```

## Dataset and leakage control

The project uses the TN5000 thyroid ultrasound dataset. The original dataset is **not included** in this repository; see [`data/README.md`](data/README.md) for the official source and reproduction instructions.

Exact pixel hashing was used to identify duplicate image groups and potential cross-split leakage. A documented Option A strategy preserved the official test set and removed duplicate TRAIN/VAL copies from the modeling data. The resulting modeling set contains **4,933 images**.

## Analytical dataset

The final analytical dataset contains **4,933 rows and 47 features**. Features cover:

- image dimensions and metadata
- nodule bounding-box geometry
- nodule area and relative size
- nodule aspect ratio and position
- whole-image intensity statistics
- whole-image entropy, contrast and edge density
- nodule-region intensity statistics
- nodule-region entropy, contrast and edge density

The analysis scripts are in `scripts/analysis/` and the dataset-construction scripts are in `scripts/preprocessing/`.

## Statistical analysis

The statistical-analysis pipeline evaluates selected features using:

- Mann–Whitney U tests
- Benjamini–Hochberg false-discovery-rate correction
- Cohen's *d*
- Cliff's delta
- bootstrap 95% confidence intervals

These analyses describe associations in the TN5000 dataset; they are not clinical diagnostic rules.

## Model development

Three EfficientNetB0 experiments are retained for comparison:

| Model | Input representation | Test ROC-AUC | Accuracy | Precision | Sensitivity | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| V1 | Full ultrasound image | 0.7379 | 0.7470 | 0.7484 | 0.9850 | 0.1004 | 0.8506 |
| V2 | Full ultrasound image | 0.8515 | 0.7520 | 0.9185 | 0.7250 | 0.8253 | 0.8104 |
| V3 | Nodule-focused crop | **0.8948** | **0.7620** | **0.9410** | **0.7196** | **0.8773** | **0.8155** |

V3 uses a validation-selected decision threshold of **0.55**. On the untouched 1,000-image test set, its confusion matrix is:

```text
                 Predicted
               Benign  Malignant
Actual Benign     236       33
Actual Malignant  205      526
```

The final metrics are stored in `results/V3/v3_final_test_metrics.json`.

## Final application

The desktop application is `scripts/application/thyroid_app.py`.

The final V3 workflow is intentionally **manual ROI selection** because the classifier was trained on nodule-focused crops; it is not an automatic nodule detector.

```text
Upload ultrasound
       ↓
Select nodule ROI manually
       ↓
Confirm ROI
       ↓
10% padding
       ↓
224×224 resize
       ↓
EfficientNetB0 preprocess_input
       ↓
V3 model
       ↓
Benign / Malignant + probability
```

## Running the application

### 1. Create an environment

Python 3.11 is recommended for the environment used during development.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start the application

From the repository root:

```bash
python scripts/application/thyroid_app.py
```

The application expects the final model at:

```text
models/V3/TN5000_EfficientNetB0_V3_NoduleCrop_final.keras
```

## Reproducing the data pipeline

After downloading TN5000 and placing it locally, the intended order is:

1. `scripts/preprocessing/tn5000_analysis.py`
2. `scripts/preprocessing/tn5000_duplicate_leakage_check.py`
3. `scripts/preprocessing/tn5000_duplicate_group_analysis.py`
4. `scripts/preprocessing/tn5000_option_a_split_planner.py`
5. `scripts/preprocessing/tn5000_create_leakage_safe_dataset.py`
6. `scripts/preprocessing/tn5000_build_analytical_dataset.py`
7. `scripts/analysis/tn5000_eda_analysis.py`
8. `scripts/analysis/tn5000_statistical_analysis.py`
9. training scripts under `scripts/training/`
10. `scripts/application/thyroid_app.py`

The scripts were originally developed on Windows and some contain absolute development paths. Those paths should be changed to local paths before reproduction.

## Repository structure

```text
Thyroid-Nodule-AI-Analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── V3/
│       └── TN5000_EfficientNetB0_V3_NoduleCrop_final.keras
├── scripts/
│   ├── preprocessing/
│   ├── analysis/
│   ├── training/
│   ├── application/
│   └── legacy/
├── results/
│   ├── V1/
│   ├── V2/
│   └── V3/
└── docs/
    ├── analysis/
    ├── architecture/
    └── screenshots/
```

## Reproducibility and limitations

- TN5000 source images and XML annotations are intentionally excluded from GitHub.
- The final classifier is a binary research model, not a clinical diagnostic system.
- V3 expects a manually selected nodule region.
- Test-set metrics are reported on the fixed 1,000-image held-out test set.
- Handcrafted analytical features are used for data analytics; the EfficientNet classifier itself learns from image crops rather than directly consuming the 47 handcrafted features.
- Exact duplicate hashing detects identical image files; it does not establish patient identity.

## Citation

If you use TN5000, cite the dataset and the associated Scientific Data article from the official source listed in [`data/README.md`](data/README.md).

## Model redistribution note

The trained model is included here as a project artifact. Before making the repository public, verify that redistribution of trained weights is compatible with the current TN5000 dataset terms and the associated publication terms. If required, keep the model outside the public repository and document how to obtain or regenerate it instead.
