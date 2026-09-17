# Dataset

This repository does **not** contain the TN5000 ultrasound images or XML annotation files.

## TN5000 dataset

- 5,000 B-mode thyroid ultrasound images
- Pascal VOC-style XML annotations
- Benign and malignant classes
- Official image-level split: 3,500 train / 500 validation / 1,000 test
- The project performs an additional exact-duplicate check before modeling.
- The modeling dataset used by the final V3 pipeline contains 4,933 retained images after the documented leakage-control procedure.

## Official source

TN5000: An Ultrasound Image Dataset for Thyroid Nodule Detection and Classification

Figshare: https://springernature.figshare.com/articles/dataset/TN5000_An_Ultrasound_Image_Dataset_for_Thyroid_Nodule_Detection_and_Classification/28455641

Dataset DOI: https://doi.org/10.6084/m9.figshare.28455641

Research article DOI: https://doi.org/10.1038/s41597-025-05757-4

## Reproduction

Download the dataset from its official source and place it locally in:

```text
project_root/data/TN5000_forReview/
├── Annotations/
├── ImageSets/Main/
└── JPEGImages/
```

Then update the dataset path in the preprocessing scripts if required. The scripts currently contain the Windows path used during development as a historical reference; for GitHub reproduction, replace it with your local path or refactor it to a project-relative path.

Do not commit the downloaded dataset to this repository unless its current license and redistribution terms explicitly permit it.
