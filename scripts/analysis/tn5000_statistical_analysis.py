import os
import csv
import math
import statistics

import numpy as np

# Optional SciPy dependency. The script gives a clear message if missing.
try:
    from scipy.stats import mannwhitneyu, shapiro
    from scipy.stats import bootstrap
    from scipy.stats import rankdata
except ImportError:
    raise SystemExit(
        "ERROR: SciPy is required.\n"
        "Install it with:\n"
        "  python -m pip install scipy"
    )

DATA_DIR = r"C:\Users\aupat\Documents\Project's\Thyroide\Thyroide_Project\data"

INPUT_CSV = os.path.join(
    DATA_DIR,
    "TN5000_Analysis",
    "thyroid_features.csv"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "TN5000_Analysis",
    "Statistical_Analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

FEATURES = [
    "nodule_area",
    "nodule_aspect_ratio",
    "nodule_area_ratio",
    "nodule_mean_intensity",
    "nodule_entropy",
    "nodule_contrast",
    "nodule_edge_density",
    "image_mean_intensity",
]

# We use the Benjamini-Hochberg procedure to control the
# false discovery rate across the multiple feature tests.
ALPHA = 0.05

# Shapiro-Wilk is useful for checking distributions, but SciPy
# warns for very large samples. We therefore inspect a random
# sample of at most 5,000 observations per class.
NORMALITY_SAMPLE_SIZE = 5000

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric_values(rows, feature):
    result = []
    for row in rows:
        try:
            value = float(row[feature])
            if np.isfinite(value):
                result.append(value)
        except (ValueError, TypeError, KeyError):
            continue
    return np.asarray(result, dtype=float)


def percentile(values, p):
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, p))


def cohens_d(benign, malignant):
    if len(benign) < 2 or len(malignant) < 2:
        return float("nan")

    n1 = len(benign)
    n2 = len(malignant)

    v1 = np.var(benign, ddof=1)
    v2 = np.var(malignant, ddof=1)

    pooled_sd = math.sqrt(
        ((n1 - 1) * v1 + (n2 - 1) * v2)
        / (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return 0.0

    return float(
        (np.mean(malignant) - np.mean(benign))
        / pooled_sd
    )


def rank_biserial_from_u(u, n_benign, n_malignant):
    """
    Rank-biserial correlation for the direction
    malignant - benign.

    r_rb = 2U/(n1*n2) - 1

    Here U is calculated for the first sample (benign),
    so positive values mean benign tends to have larger
    observations and negative values mean malignant tends
    to have larger observations.
    """
    if n_benign == 0 or n_malignant == 0:
        return float("nan")

    return float(
        (2.0 * u) / (n_benign * n_malignant) - 1.0
    )


def cliffs_delta(benign, malignant):
    """
    Approximate Cliff's delta using ranks.

    Positive delta => malignant values tend to be larger.
    Negative delta => benign values tend to be larger.
    """
    n1 = len(benign)
    n2 = len(malignant)

    if n1 == 0 or n2 == 0:
        return float("nan")

    combined = np.concatenate([benign, malignant])
    ranks = rankdata(combined, method="average")

    r_b = np.sum(ranks[:n1])

    # Mann-Whitney U for benign as first group
    u_b = r_b - n1 * (n1 + 1) / 2.0

    # U_b counts benign > malignant (with ties half).
    # Convert to malignant > benign direction.
    u_m = n1 * n2 - u_b

    return float(
        (2.0 * u_m) / (n1 * n2) - 1.0
    )


def bh_fdr(p_values):
    """
    Benjamini-Hochberg adjusted p-values.
    Returns adjusted values in original order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    previous = 1.0

    for i in range(m - 1, -1, -1):
        idx = order[i]
        rank = i + 1

        value = p[idx] * m / rank
        value = min(value, previous)

        adjusted[idx] = value
        previous = value

    return adjusted


def bootstrap_mean_difference_ci(benign, malignant, iterations=2000):
    """
    Bootstrap 95% CI for:
        malignant mean - benign mean
    """
    rng = np.random.default_rng(RANDOM_SEED)

    n_b = len(benign)
    n_m = len(malignant)

    if n_b < 2 or n_m < 2:
        return float("nan"), float("nan")

    # To keep runtime reasonable, calculate in chunks.
    diffs = np.empty(iterations, dtype=float)

    chunk = 100
    pos = 0

    while pos < iterations:
        current = min(chunk, iterations - pos)

        b_idx = rng.integers(0, n_b, size=(current, n_b))
        m_idx = rng.integers(0, n_m, size=(current, n_m))

        b_means = np.mean(benign[b_idx], axis=1)
        m_means = np.mean(malignant[m_idx], axis=1)

        diffs[pos:pos + current] = m_means - b_means
        pos += current

    return (
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5))
    )


def significance_label(p_adj):
    if p_adj < 0.001:
        return "Highly significant"
    if p_adj < 0.01:
        return "Very significant"
    if p_adj < 0.05:
        return "Significant"
    return "Not significant"


def effect_strength(d):
    a = abs(d)

    if a < 0.2:
        return "Negligible / very small"
    if a < 0.5:
        return "Small"
    if a < 0.8:
        return "Medium"
    return "Large"


# ============================================================
# START
# ============================================================

print("=" * 76)
print("TN5000 - STATISTICAL VALIDATION OF EDA FINDINGS")
print("=" * 76)
print("Input :", INPUT_CSV)
print("Output:", OUTPUT_DIR)
print()

if not os.path.isfile(INPUT_CSV):
    raise SystemExit(
        "ERROR: thyroid_features.csv was not found."
    )

rows = read_csv(INPUT_CSV)

if not rows:
    raise SystemExit(
        "ERROR: CSV contains no rows."
    )

benign_rows = [
    r for r in rows
    if r.get("class") == "Benign"
]

malignant_rows = [
    r for r in rows
    if r.get("class") == "Malignant"
]

print("[1] DATASET")
print("Total rows:", len(rows))
print("Benign:", len(benign_rows))
print("Malignant:", len(malignant_rows))
print()

print("[2] TESTING METHOD")
print("Primary test: Mann-Whitney U")
print("Multiple-testing correction: Benjamini-Hochberg FDR")
print("Alpha:", ALPHA)
print("Effect sizes: Cohen's d + Cliff's delta")
print("95% CI: bootstrap CI for mean difference")
print()

results = []

for feature in FEATURES:

    benign = numeric_values(benign_rows, feature)
    malignant = numeric_values(malignant_rows, feature)

    if len(benign) == 0 or len(malignant) == 0:
        continue

    # Mann-Whitney U
    u_result = mannwhitneyu(
        benign,
        malignant,
        alternative="two-sided"
    )

    u_stat = float(u_result.statistic)
    p_value = float(u_result.pvalue)

    # Descriptive statistics
    benign_mean = float(np.mean(benign))
    malignant_mean = float(np.mean(malignant))

    benign_median = float(np.median(benign))
    malignant_median = float(np.median(malignant))

    mean_difference = malignant_mean - benign_mean
    median_difference = malignant_median - benign_median

    # Effect sizes
    d = cohens_d(benign, malignant)
    cliff = cliffs_delta(benign, malignant)

    # Rank-biserial in malignant direction
    rank_biserial = rank_biserial_from_u(
        u_stat,
        len(benign),
        len(malignant)
    )

    # Bootstrap confidence interval
    ci_low, ci_high = bootstrap_mean_difference_ci(
        benign,
        malignant
    )

    # Normality checks
    rng = np.random.default_rng(RANDOM_SEED)

    b_sample = benign
    m_sample = malignant

    if len(b_sample) > NORMALITY_SAMPLE_SIZE:
        b_sample = rng.choice(
            b_sample,
            NORMALITY_SAMPLE_SIZE,
            replace=False
        )

    if len(m_sample) > NORMALITY_SAMPLE_SIZE:
        m_sample = rng.choice(
            m_sample,
            NORMALITY_SAMPLE_SIZE,
            replace=False
        )

    # Shapiro is not the decision-maker for the Mann-Whitney test;
    # it is included for distribution description only.
    b_shapiro = shapiro(b_sample)
    m_shapiro = shapiro(m_sample)

    results.append({
        "feature": feature,
        "benign_n": len(benign),
        "malignant_n": len(malignant),
        "benign_mean": benign_mean,
        "malignant_mean": malignant_mean,
        "mean_difference_malignant_minus_benign": mean_difference,
        "benign_median": benign_median,
        "malignant_median": malignant_median,
        "median_difference_malignant_minus_benign": median_difference,
        "benign_p25": percentile(benign, 25),
        "benign_p75": percentile(benign, 75),
        "malignant_p25": percentile(malignant, 25),
        "malignant_p75": percentile(malignant, 75),
        "mann_whitney_u": u_stat,
        "p_value": p_value,
        "cohens_d": d,
        "absolute_cohens_d": abs(d),
        "cliffs_delta": cliff,
        "rank_biserial": rank_biserial,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "benign_shapiro_p": float(b_shapiro.pvalue),
        "malignant_shapiro_p": float(m_shapiro.pvalue),
    })

p_values = [r["p_value"] for r in results]
adjusted = bh_fdr(p_values)

for r, p_adj in zip(results, adjusted):
    r["p_value_fdr_bh"] = float(p_adj)
    r["significance"] = significance_label(p_adj)
    r["effect_strength"] = effect_strength(r["cohens_d"])

    # Direction based on mean difference.
    if r["mean_difference_malignant_minus_benign"] > 0:
        r["direction"] = "Higher in malignant"
    elif r["mean_difference_malignant_minus_benign"] < 0:
        r["direction"] = "Lower in malignant"
    else:
        r["direction"] = "No mean difference"

# Rank by adjusted p-value, then absolute effect size.
results.sort(
    key=lambda r: (
        r["p_value_fdr_bh"],
        -r["absolute_cohens_d"]
    )
)

print("[3] RESULTS")
print()

for i, r in enumerate(results, start=1):

    print(
        f"{i:02d}. {r['feature']}"
    )
    print(
        f"    Benign mean     : {r['benign_mean']:.6f}"
    )
    print(
        f"    Malignant mean  : {r['malignant_mean']:.6f}"
    )
    print(
        f"    Mean difference : "
        f"{r['mean_difference_malignant_minus_benign']:.6f}"
    )
    print(
        f"    U statistic     : {r['mann_whitney_u']:.4f}"
    )
    print(
        f"    p-value         : {r['p_value']:.8g}"
    )
    print(
        f"    FDR-adjusted p  : {r['p_value_fdr_bh']:.8g}"
    )
    print(
        f"    Cohen's d       : {r['cohens_d']:.6f}"
    )
    print(
        f"    Cliff's delta   : {r['cliffs_delta']:.6f}"
    )
    print(
        f"    95% CI          : "
        f"[{r['bootstrap_ci_low']:.6f}, "
        f"{r['bootstrap_ci_high']:.6f}]"
    )
    print(
        f"    Result          : "
        f"{r['significance']} | "
        f"{r['direction']}"
    )
    print()

# ============================================================
# SAVE COMPLETE RESULTS
# ============================================================

results_csv = os.path.join(
    OUTPUT_DIR,
    "statistical_feature_tests.csv"
)

fieldnames = list(results[0].keys())

with open(
    results_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )
    writer.writeheader()
    writer.writerows(results)

# ============================================================
# SIGNIFICANT FEATURES ONLY
# ============================================================

significant = [
    r for r in results
    if r["p_value_fdr_bh"] < ALPHA
]

significant_csv = os.path.join(
    OUTPUT_DIR,
    "statistically_supported_features.csv"
)

with open(
    significant_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )
    writer.writeheader()
    writer.writerows(significant)

# ============================================================
# REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "statistical_analysis_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("TN5000 STATISTICAL ANALYSIS REPORT\n")
    f.write("=" * 76 + "\n\n")

    f.write("DATASET\n")
    f.write("-" * 76 + "\n")
    f.write(f"Rows analysed: {len(rows)}\n")
    f.write(f"Benign: {len(benign_rows)}\n")
    f.write(f"Malignant: {len(malignant_rows)}\n")
    f.write(f"Features tested: {len(results)}\n\n")

    f.write("METHODS\n")
    f.write("-" * 76 + "\n")
    f.write(
        "Mann-Whitney U test was used as the primary two-group "
        "comparison because image-derived features may not follow "
        "normal distributions.\n"
    )
    f.write(
        "Benjamini-Hochberg false-discovery-rate correction was "
        "applied across the tested features.\n"
    )
    f.write(
        "Cohen's d and Cliff's delta were calculated as effect-size "
        "measures.\n"
    )
    f.write(
        "A bootstrap 95% confidence interval was calculated for the "
        "malignant-minus-benign mean difference.\n"
    )
    f.write(
        "Statistical significance threshold: adjusted p < 0.05.\n\n"
    )

    f.write("FEATURE RESULTS\n")
    f.write("-" * 76 + "\n\n")

    for i, r in enumerate(results, start=1):
        f.write(
            f"{i}. {r['feature']}\n"
            f"   Benign mean: {r['benign_mean']:.6f}\n"
            f"   Malignant mean: {r['malignant_mean']:.6f}\n"
            f"   Mean difference (M-B): "
            f"{r['mean_difference_malignant_minus_benign']:.6f}\n"
            f"   Mann-Whitney U: {r['mann_whitney_u']:.6f}\n"
            f"   Raw p-value: {r['p_value']:.10g}\n"
            f"   FDR-adjusted p-value: "
            f"{r['p_value_fdr_bh']:.10g}\n"
            f"   Cohen's d: {r['cohens_d']:.6f}\n"
            f"   Cliff's delta: {r['cliffs_delta']:.6f}\n"
            f"   95% CI: "
            f"[{r['bootstrap_ci_low']:.6f}, "
            f"{r['bootstrap_ci_high']:.6f}]\n"
            f"   Direction: {r['direction']}\n"
            f"   Significance: {r['significance']}\n"
            f"   Effect strength: {r['effect_strength']}\n\n"
        )

    f.write("STATISTICALLY SUPPORTED FEATURES\n")
    f.write("-" * 76 + "\n")

    if significant:
        for r in significant:
            f.write(
                f"- {r['feature']}: "
                f"adjusted p={r['p_value_fdr_bh']:.10g}, "
                f"Cohen's d={r['cohens_d']:.6f}\n"
            )
    else:
        f.write(
            "No feature passed the adjusted p < 0.05 threshold.\n"
        )

    f.write("\nINTERPRETATION NOTES\n")
    f.write("-" * 76 + "\n")
    f.write(
        "A statistically significant difference means the feature "
        "distributions differ between the two labelled groups in "
        "this dataset. It does not establish causation, clinical "
        "diagnostic validity, or that the feature alone can classify "
        "a new patient reliably.\n"
    )
    f.write(
        "The statistical analysis is descriptive/exploratory and is "
        "separate from the final AI model evaluation.\n"
    )

print("=" * 76)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 76)
print("Complete results :", results_csv)
print("Supported features:", significant_csv)
print("Report           :", report_path)
print()
print("Statistically supported features:", len(significant))
print("No source data was modified.")
