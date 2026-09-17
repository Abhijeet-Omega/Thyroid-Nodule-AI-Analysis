import os

# Force CPU-only execution BEFORE importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ============================================================
# TN5000 THYROID ULTRASOUND AI
# CPU-ONLY EfficientNetB0 Training Pipeline
# ============================================================

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_DIR = "/mnt/c/Users/aupat/Documents/Project's/Thyroide/Thyroide_Project/data/TN5000_LeakageSafe"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = os.path.expanduser("~/thyroid_ai/results")

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42

# CPU-friendly training
FROZEN_EPOCHS = 5
FINE_TUNE_EPOCHS = 5

INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5

# ------------------------------------------------------------
# CPU configuration
# ------------------------------------------------------------

tf.keras.utils.set_random_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("TN5000 THYROID ULTRASOUND AI")
print("CPU-ONLY EfficientNetB0 TRAINING")
print("=" * 70)

print("\nTensorFlow version:", tf.__version__)
print("CPU devices:", tf.config.list_physical_devices("CPU"))
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# ------------------------------------------------------------
# Verify dataset
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. DATASET VERIFICATION")
print("=" * 70)

for name, path in [
    ("Train", TRAIN_DIR),
    ("Validation", VAL_DIR),
    ("Test", TEST_DIR),
]:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing {name} directory: {path}")

    print(f"{name:12}: OK")

# ------------------------------------------------------------
# Count class images
# ------------------------------------------------------------

def count_class_images(directory):
    counts = {}

    for class_name in ["benign", "malignant"]:
        class_dir = os.path.join(directory, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(class_dir)

        count = 0

        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1

        counts[class_name] = count

    return counts


train_counts = count_class_images(TRAIN_DIR)
val_counts = count_class_images(VAL_DIR)
test_counts = count_class_images(TEST_DIR)

print("\nClass distribution:")

print("\nTRAIN")
print("  Benign    :", train_counts["benign"])
print("  Malignant :", train_counts["malignant"])
print("  Total     :", sum(train_counts.values()))

print("\nVALIDATION")
print("  Benign    :", val_counts["benign"])
print("  Malignant :", val_counts["malignant"])
print("  Total     :", sum(val_counts.values()))

print("\nTEST")
print("  Benign    :", test_counts["benign"])
print("  Malignant :", test_counts["malignant"])
print("  Total     :", sum(test_counts.values()))

# ------------------------------------------------------------
# Create datasets
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. LOADING IMAGE DATASETS")
print("=" * 70)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("\nClass names:", train_ds.class_names)

if train_ds.class_names != ["benign", "malignant"]:
    raise RuntimeError(
        f"Unexpected class ordering: {train_ds.class_names}"
    )

print("Class mapping:")
print("  0 = benign")
print("  1 = malignant")

# ------------------------------------------------------------
# Performance pipeline
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ------------------------------------------------------------
# Class weights
# ------------------------------------------------------------

# Weight the minority benign class more heavily.
total_train = train_counts["benign"] + train_counts["malignant"]

class_weight = {
    0: total_train / (2 * train_counts["benign"]),
    1: total_train / (2 * train_counts["malignant"]),
}

print("\nClass weights:")
print("  Benign    :", round(class_weight[0], 4))
print("  Malignant :", round(class_weight[1], 4))

# ------------------------------------------------------------
# Data augmentation
# ------------------------------------------------------------

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="data_augmentation",
)

# ------------------------------------------------------------
# Build EfficientNetB0
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. BUILDING EFFICIENTNETB0")
print("=" * 70)

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

base_model.trainable = False

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="ultrasound_image"
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
)(x)

x = tf.keras.layers.Dropout(0.40)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="malignancy_probability"
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="TN5000_EfficientNetB0"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LR
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

print("\nModel created successfully.")

print(
    "Total parameters:",
    f"{model.count_params():,}"
)

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "best_frozen_model.keras"
)

callbacks_frozen = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1,
    ),
]

# ------------------------------------------------------------
# Phase 1: Frozen EfficientNet
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. PHASE 1 — TRANSFER LEARNING")
print("=" * 70)

print("EfficientNetB0 base: FROZEN")
print("Epochs:", FROZEN_EPOCHS)
print("Learning rate:", INITIAL_LR)

start_time = time.time()

history_frozen = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FROZEN_EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks_frozen,
    verbose=1,
)

frozen_time = time.time() - start_time

print(
    "\nPhase 1 training time:",
    round(frozen_time / 60, 2),
    "minutes"
)

# ------------------------------------------------------------
# Phase 2: Fine tuning
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. PHASE 2 — FINE TUNING")
print("=" * 70)

# Unfreeze the top portion of EfficientNetB0.
base_model.trainable = True

# Keep most of the pretrained network frozen.
fine_tune_start = max(
    0,
    len(base_model.layers) - 40
)

for layer in base_model.layers[:fine_tune_start]:
    layer.trainable = False

for layer in base_model.layers[fine_tune_start:]:
    layer.trainable = True

# Keep BatchNormalization layers frozen for stable transfer learning.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(
    1 for layer in model.layers
    if layer.trainable
)

print("Fine-tuning top EfficientNetB0 layers.")
print("Trainable model layers:", trainable_count)
print("Fine-tuning learning rate:", FINE_TUNE_LR)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LR
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

checkpoint_finetuned = os.path.join(
    OUTPUT_DIR,
    "best_finetuned_model.keras"
)

callbacks_finetune = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_finetuned,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-8,
        verbose=1,
    ),
]

start_time = time.time()

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks_finetune,
    verbose=1,
)

finetune_time = time.time() - start_time

print(
    "\nPhase 2 training time:",
    round(finetune_time / 60, 2),
    "minutes"
)

# ------------------------------------------------------------
# Save final model
# ------------------------------------------------------------

final_model_path = os.path.join(
    OUTPUT_DIR,
    "TN5000_EfficientNetB0_final.keras"
)

model.save(final_model_path)

print("\nFinal model saved:")
print(final_model_path)

# ------------------------------------------------------------
# Combine training history
# ------------------------------------------------------------

history = {}

for key in history_frozen.history:
    history[key] = (
        history_frozen.history[key]
        + history_finetune.history.get(key, [])
    )

history_df = pd.DataFrame(history)

history_csv = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)

history_df.to_csv(
    history_csv,
    index=False
)

# ------------------------------------------------------------
# Plot training history
# ------------------------------------------------------------

epochs_range = range(1, len(history_df) + 1)

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_range,
    history_df["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    history_df["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("TN5000 Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

accuracy_plot = os.path.join(
    OUTPUT_DIR,
    "training_accuracy.png"
)

plt.savefig(
    accuracy_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_range,
    history_df["loss"],
    label="Training Loss"
)

plt.plot(
    epochs_range,
    history_df["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("TN5000 Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)

loss_plot = os.path.join(
    OUTPUT_DIR,
    "training_loss.png"
)

plt.savefig(
    loss_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Prediction helper
# ------------------------------------------------------------

def get_predictions(dataset):
    probabilities = model.predict(
        dataset,
        verbose=1
    )

    probabilities = probabilities.reshape(-1)

    labels = np.concatenate(
        [
            y.numpy().reshape(-1)
            for _, y in dataset
        ]
    ).astype(int)

    return labels, probabilities


# ------------------------------------------------------------
# Validation predictions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. VALIDATION THRESHOLD SELECTION")
print("=" * 70)

val_labels, val_probs = get_predictions(val_ds)

val_auc = roc_auc_score(
    val_labels,
    val_probs
)

print("Validation ROC-AUC:", round(val_auc, 5))

# Search threshold using validation F1.
thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

threshold_results = []

for threshold in thresholds:

    predictions = (
        val_probs >= threshold
    ).astype(int)

    threshold_results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy_score(
                val_labels,
                predictions
            ),
            "precision": precision_score(
                val_labels,
                predictions,
                zero_division=0
            ),
            "recall": recall_score(
                val_labels,
                predictions,
                zero_division=0
            ),
            "f1": f1_score(
                val_labels,
                predictions,
                zero_division=0
            ),
        }
    )

threshold_df = pd.DataFrame(
    threshold_results
)

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

BEST_THRESHOLD = float(
    best_row["threshold"]
)

print("\nSelected threshold:")
print("Threshold:", round(BEST_THRESHOLD, 2))
print("Validation accuracy:", round(best_row["accuracy"], 4))
print("Validation precision:", round(best_row["precision"], 4))
print("Validation recall:", round(best_row["recall"], 4))
print("Validation F1:", round(best_row["f1"], 4))

threshold_csv = os.path.join(
    OUTPUT_DIR,
    "validation_threshold_results.csv"
)

threshold_df.to_csv(
    threshold_csv,
    index=False
)

# ------------------------------------------------------------
# Final untouched test evaluation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("7. FINAL TEST EVALUATION")
print("=" * 70)

print("IMPORTANT:")
print("The test set has not been used for training")
print("or threshold selection.")

test_labels, test_probs = get_predictions(test_ds)

test_predictions = (
    test_probs >= BEST_THRESHOLD
).astype(int)

# Metrics
accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

specificity = recall_score(
    test_labels,
    test_predictions,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    test_labels,
    test_probs
)

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print("\nFINAL TEST RESULTS")
print("-" * 50)

print("Threshold   :", round(BEST_THRESHOLD, 4))
print("Accuracy    :", round(accuracy, 4))
print("Precision   :", round(precision, 4))
print("Sensitivity :", round(recall, 4))
print("Specificity :", round(specificity, 4))
print("F1 Score    :", round(f1, 4))
print("ROC-AUC     :", round(test_auc, 4))

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=["Benign", "Malignant"],
        digits=4,
        zero_division=0
    )
)

# ------------------------------------------------------------
# Save metrics
# ------------------------------------------------------------

metrics = {
    "threshold": BEST_THRESHOLD,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "sensitivity_recall": float(recall),
    "specificity": float(specificity),
    "f1_score": float(f1),
    "roc_auc": float(test_auc),
    "true_negative": int(cm[0, 0]),
    "false_positive": int(cm[0, 1]),
    "false_negative": int(cm[1, 0]),
    "true_positive": int(cm[1, 1]),
    "train_images": int(sum(train_counts.values())),
    "validation_images": int(sum(val_counts.values())),
    "test_images": int(sum(test_counts.values())),
}

metrics_path = os.path.join(
    OUTPUT_DIR,
    "final_test_metrics.json"
)

with open(metrics_path, "w") as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

# ------------------------------------------------------------
# Save test predictions
# ------------------------------------------------------------

test_prediction_df = pd.DataFrame(
    {
        "actual_class": test_labels,
        "malignant_probability": test_probs,
        "predicted_class": test_predictions,
    }
)

test_predictions_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

test_prediction_df.to_csv(
    test_predictions_path,
    index=False
)

# ------------------------------------------------------------
# ROC curve
# ------------------------------------------------------------

fpr, tpr, roc_thresholds = roc_curve(
    test_labels,
    test_probs
)

plt.figure(figsize=(8, 7))

plt.plot(
    fpr,
    tpr,
    label=f"EfficientNetB0 (AUC = {test_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("TN5000 ROC Curve — Test Set")
plt.legend()
plt.grid(True, alpha=0.3)

roc_path = os.path.join(
    OUTPUT_DIR,
    "test_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Confusion matrix plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("TN5000 Test Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(2)

plt.xticks(
    tick_marks,
    ["Benign", "Malignant"]
)

plt.yticks(
    tick_marks,
    ["Benign", "Malignant"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "test_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Misclassified test images
# ------------------------------------------------------------

test_file_paths = []

for class_name in ["benign", "malignant"]:

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    for filename in sorted(
        os.listdir(class_dir)
    ):
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            test_file_paths.append(
                os.path.join(
                    class_dir,
                    filename
                )
            )

# image_dataset_from_directory uses sorted class-directory
# ordering and shuffle=False, so this ordering corresponds
# to the prediction ordering.

misclassified = []

for path, actual, probability, predicted in zip(
    test_file_paths,
    test_labels,
    test_probs,
    test_predictions
):

    if actual != predicted:

        actual_name = (
            "malignant"
            if actual == 1
            else "benign"
        )

        predicted_name = (
            "malignant"
            if predicted == 1
            else "benign"
        )

        misclassified.append(
            {
                "image_path": path,
                "actual": actual_name,
                "predicted": predicted_name,
                "malignant_probability": probability,
            }
        )

misclassified_df = pd.DataFrame(
    misclassified
)

misclassified_path = os.path.join(
    OUTPUT_DIR,
    "misclassified_test_images.csv"
)

misclassified_df.to_csv(
    misclassified_path,
    index=False
)

# ------------------------------------------------------------
# Save training configuration
# ------------------------------------------------------------

config = {
    "dataset": "TN5000 Leakage-Safe",
    "train_images": sum(train_counts.values()),
    "validation_images": sum(val_counts.values()),
    "test_images": sum(test_counts.values()),
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "model": "EfficientNetB0",
    "frozen_epochs": FROZEN_EPOCHS,
    "fine_tune_epochs": FINE_TUNE_EPOCHS,
    "initial_learning_rate": INITIAL_LR,
    "fine_tune_learning_rate": FINE_TUNE_LR,
    "class_mapping": {
        "0": "benign",
        "1": "malignant"
    },
    "threshold_selection": "Validation F1 optimization",
    "test_set_used_for_training": False,
}

config_path = os.path.join(
    OUTPUT_DIR,
    "training_configuration.json"
)

with open(config_path, "w") as f:
    json.dump(
        config,
        f,
        indent=4
    )

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING AND EVALUATION COMPLETE")
print("=" * 70)

print("\nFinal Test Metrics")
print("-" * 50)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Sensitivity : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"ROC-AUC     : {test_auc:.4f}")

print("\nSelected threshold:", round(BEST_THRESHOLD, 4))

print("\nResults directory:")
print(OUTPUT_DIR)

print("\nImportant files:")
print("  Final model       :", final_model_path)
print("  Test metrics      :", metrics_path)
print("  Test predictions  :", test_predictions_path)
print("  ROC curve         :", roc_path)
print("  Confusion matrix  :", cm_path)
print("  Training history  :", history_csv)

print("\nAcademic note:")
print(
    "This model is an academic decision-support prototype "
    "and is not a clinical diagnostic system."
)

print("=" * 70)
