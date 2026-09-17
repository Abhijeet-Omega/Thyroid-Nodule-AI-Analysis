import os
import csv
import math
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# TN5000 - EDA / DATA ANALYTICS REPORT
#
# Input:
#   data/TN5000_Analysis/thyroid_features.csv
#
# Outputs:
#   data/TN5000_Analysis/EDA/
#       01_class_distribution.png
#       02_nodule_area_by_class.png
#       03_nodule_aspect_ratio_by_class.png
#       04_nodule_area_ratio_by_class.png
#       05_nodule_intensity_by_class.png
#       06_nodule_entropy_by_class.png
#       07_nodule_contrast_by_class.png
#       08_nodule_edge_density_by_class.png
#       09_image_intensity_by_class.png
#       10_feature_correlation.csv
#       11_feature_class_statistics.csv
#       12_feature_effect_size.csv
#       EDA_summary.txt
#
# No source data is modified.
# ============================================================

DATA_DIR = r"C:\Users\aupat\Documents\Project's\Thyroide\Thyroide_Project\data"
CSV_PATH = os.path.join(
    DATA_DIR,
    "TN5000_Analysis",
    "thyroid_features.csv"
)

OUT_DIR = os.path.join(
    DATA_DIR,
    "TN5000_Analysis",
    "EDA"
)

os.makedirs(OUT_DIR, exist_ok=True)


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(rows, column):
    values = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (ValueError, TypeError):
            pass
    return np.asarray(values, dtype=float)


def mean_median_std(values):
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.std(values))
    )


def percentile(values, p):
    return float(np.percentile(values, p)) if len(values) else 0.0


def cohens_d(a, b):
    """
    Cohen's d using pooled standard deviation.
    Positive d means malignant > benign.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0

    ma = np.mean(a)
    mb = np.mean(b)

    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)

    pooled = math.sqrt(
        ((len(a) - 1) * va + (len(b) - 1) * vb)
        / (len(a) + len(b) - 2)
    )

    if pooled == 0:
        return 0.0

    return float((mb - ma) / pooled)


def save_bar_chart(labels, values, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, filename),
        dpi=180
    )
    plt.close()


def save_boxplot(benign, malignant, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [benign, malignant],
        labels=["Benign", "Malignant"],
        showfliers=False
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, filename),
        dpi=180
    )
    plt.close()


def save_histogram(benign, malignant, title, xlabel, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(
        benign,
        bins=40,
        alpha=0.55,
        label="Benign"
    )
    plt.hist(
        malignant,
        bins=40,
        alpha=0.55,
        label="Malignant"
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, filename),
        dpi=180
    )
    plt.close()


print("=" * 72)
print("TN5000 - EXPLORATORY DATA ANALYSIS")
print("=" * 72)
print("Input:", CSV_PATH)
print("Output:", OUT_DIR)
print()

if not os.path.isfile(CSV_PATH):
    raise SystemExit(
        "ERROR: thyroid_features.csv was not found."
    )

rows = read_csv(CSV_PATH)

if not rows:
    raise SystemExit(
        "ERROR: thyroid_features.csv contains no data."
    )

print("[1] DATASET OVERVIEW")

total = len(rows)
benign_rows = [
    r for r in rows
    if r["class"] == "Benign"
]
malignant_rows = [
    r for r in rows
    if r["class"] == "Malignant"
]

print("Total rows:", total)
print("Features:", len(rows[0]))
print("Benign:", len(benign_rows))
print("Malignant:", len(malignant_rows))

print()
print("Split counts:")

split_counts = Counter(r["split"] for r in rows)

for split in ("train", "val", "test"):
    subset = [r for r in rows if r["split"] == split]
    b = sum(r["class"] == "Benign" for r in subset)
    m = sum(r["class"] == "Malignant" for r in subset)
    print(
        f"  {split}: {len(subset)} "
        f"(Benign={b}, Malignant={m})"
    )

# ------------------------------------------------------------
# Class distribution
# ------------------------------------------------------------
save_bar_chart(
    ["Benign", "Malignant"],
    [len(benign_rows), len(malignant_rows)],
    "TN5000 Class Distribution",
    "Number of images",
    "01_class_distribution.png"
)

# ------------------------------------------------------------
# Features selected for first EDA
# ------------------------------------------------------------
features = [
    ("nodule_area", "Nodule Bounding-Box Area", "Area (pixels²)",
     "02_nodule_area_by_class.png"),
    ("nodule_aspect_ratio", "Nodule Aspect Ratio", "Width / Height",
     "03_nodule_aspect_ratio_by_class.png"),
    ("nodule_area_ratio", "Nodule-to-Image Area Ratio", "Ratio",
     "04_nodule_area_ratio_by_class.png"),
    ("nodule_mean_intensity", "Nodule Mean Intensity", "Grayscale intensity",
     "05_nodule_intensity_by_class.png"),
    ("nodule_entropy", "Nodule Entropy", "Entropy (bits)",
     "06_nodule_entropy_by_class.png"),
    ("nodule_contrast", "Nodule Contrast", "RMS contrast",
     "07_nodule_contrast_by_class.png"),
    ("nodule_edge_density", "Nodule Edge Density", "Edge-density ratio",
     "08_nodule_edge_density_by_class.png"),
    ("image_mean_intensity", "Whole-Image Mean Intensity",
     "Grayscale intensity",
     "09_image_intensity_by_class.png"),
]

print()
print("[2] CLASS-WISE FEATURE ANALYSIS")

stats_rows = []
effect_rows = []

for feature, title, ylabel, filename in features:

    benign = to_float(benign_rows, feature)
    malignant = to_float(malignant_rows, feature)

    save_boxplot(
        benign,
        malignant,
        title + " — Benign vs Malignant",
        ylabel,
        filename
    )

    bm, bmed, bstd = mean_median_std(benign)
    mm, mmed, mstd = mean_median_std(malignant)

    stats_rows.append({
        "feature": feature,
        "benign_n": len(benign),
        "benign_mean": bm,
        "benign_median": bmed,
        "benign_std": bstd,
        "benign_p25": percentile(benign, 25),
        "benign_p75": percentile(benign, 75),
        "malignant_n": len(malignant),
        "malignant_mean": mm,
        "malignant_median": mmed,
        "malignant_std": mstd,
        "malignant_p25": percentile(malignant, 25),
        "malignant_p75": percentile(malignant, 75),
    })

    d = cohens_d(benign, malignant)

    effect_rows.append({
        "feature": feature,
        "cohens_d_malignant_minus_benign": d,
        "absolute_cohens_d": abs(d),
        "benign_mean": bm,
        "malignant_mean": mm,
        "mean_difference_malignant_minus_benign": mm - bm
    })

    print(
        f"{feature}: "
        f"Benign mean={bm:.4f}, "
        f"Malignant mean={mm:.4f}, "
        f"Cohen's d={d:.4f}"
    )

# ------------------------------------------------------------
# Histograms for key variables
# ------------------------------------------------------------
save_histogram(
    to_float(benign_rows, "nodule_area"),
    to_float(malignant_rows, "nodule_area"),
    "Nodule Area Distribution",
    "Nodule area (pixels²)",
    "02_nodule_area_histogram.png"
)

save_histogram(
    to_float(benign_rows, "nodule_entropy"),
    to_float(malignant_rows, "nodule_entropy"),
    "Nodule Entropy Distribution",
    "Entropy (bits)",
    "06_nodule_entropy_histogram.png"
)

# ------------------------------------------------------------
# Feature correlation matrix
# ------------------------------------------------------------
print()
print("[3] CORRELATION ANALYSIS")

correlation_features = [
    "nodule_width",
    "nodule_height",
    "nodule_area",
    "nodule_area_ratio",
    "nodule_aspect_ratio",
    "nodule_relative_width",
    "nodule_relative_height",
    "nodule_center_x_ratio",
    "nodule_center_y_ratio",
    "image_mean_intensity",
    "image_std_intensity",
    "image_entropy",
    "image_contrast",
    "image_edge_density",
    "nodule_mean_intensity",
    "nodule_std_intensity",
    "nodule_entropy",
    "nodule_contrast",
    "nodule_edge_density",
]

matrix = []

for feature in correlation_features:
    matrix.append(to_float(rows, feature))

matrix = np.asarray(matrix, dtype=float)

corr = np.corrcoef(matrix)

correlation_csv = os.path.join(
    OUT_DIR,
    "10_feature_correlation.csv"
)

with open(
    correlation_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["feature"] + correlation_features)

    for i, feature in enumerate(correlation_features):
        writer.writerow(
            [feature]
            + [float(x) for x in corr[i]]
        )

# ------------------------------------------------------------
# Save statistics CSV
# ------------------------------------------------------------
stats_csv = os.path.join(
    OUT_DIR,
    "11_feature_class_statistics.csv"
)

with open(
    stats_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    fieldnames = list(stats_rows[0].keys())
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )
    writer.writeheader()
    writer.writerows(stats_rows)

effect_csv = os.path.join(
    OUT_DIR,
    "12_feature_effect_size.csv"
)

effect_rows.sort(
    key=lambda x: x["absolute_cohens_d"],
    reverse=True
)

with open(
    effect_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    fieldnames = list(effect_rows[0].keys())
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )
    writer.writeheader()
    writer.writerows(effect_rows)

print()
print("Top features by absolute Cohen's d:")

for row in effect_rows[:10]:
    print(
        f"  {row['feature']}: "
        f"d={row['cohens_d_malignant_minus_benign']:.4f}"
    )

# ------------------------------------------------------------
# Summary report
# ------------------------------------------------------------
summary_path = os.path.join(
    OUT_DIR,
    "EDA_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("TN5000 EXPLORATORY DATA ANALYSIS SUMMARY\n")
    f.write("=" * 72 + "\n\n")

    f.write("DATASET\n")
    f.write("-" * 72 + "\n")
    f.write(f"Total analytical rows: {total}\n")
    f.write(f"Number of attributes: {len(rows[0])}\n")
    f.write(f"Benign: {len(benign_rows)}\n")
    f.write(f"Malignant: {len(malignant_rows)}\n\n")

    f.write("SPLITS\n")
    f.write("-" * 72 + "\n")

    for split in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == split]
        b = sum(r["class"] == "Benign" for r in subset)
        m = sum(r["class"] == "Malignant" for r in subset)
        f.write(
            f"{split}: {len(subset)} "
            f"(Benign={b}, Malignant={m})\n"
        )

    f.write("\nCLASS-WISE FEATURE STATISTICS\n")
    f.write("-" * 72 + "\n")

    for row in stats_rows:
        f.write(
            f"\n{row['feature']}\n"
            f"  Benign: "
            f"mean={row['benign_mean']:.6f}, "
            f"median={row['benign_median']:.6f}, "
            f"std={row['benign_std']:.6f}\n"
            f"  Malignant: "
            f"mean={row['malignant_mean']:.6f}, "
            f"median={row['malignant_median']:.6f}, "
            f"std={row['malignant_std']:.6f}\n"
        )

    f.write("\nFEATURE EFFECT SIZES\n")
    f.write("-" * 72 + "\n")
    f.write(
        "Cohen's d is reported as malignant mean minus benign mean.\n"
        "A larger absolute value indicates stronger separation in this\n"
        "simple descriptive comparison; it does NOT prove causation or\n"
        "clinical diagnostic usefulness.\n\n"
    )

    for row in effect_rows:
        f.write(
            f"{row['feature']}: "
            f"d={row['cohens_d_malignant_minus_benign']:.6f}\n"
        )

    f.write("\nOUTPUT FILES\n")
    f.write("-" * 72 + "\n")
    f.write("Class distribution and feature plots: PNG files in this folder.\n")
    f.write("10_feature_correlation.csv\n")
    f.write("11_feature_class_statistics.csv\n")
    f.write("12_feature_effect_size.csv\n")
    f.write("EDA_summary.txt\n")

print()
print("=" * 72)
print("EDA COMPLETE")
print("=" * 72)
print("Summary:", summary_path)
print("Correlation:", correlation_csv)
print("Statistics:", stats_csv)
print("Effect sizes:", effect_csv)
print("Plots:", OUT_DIR)
print()
print("No source data was modified.")
