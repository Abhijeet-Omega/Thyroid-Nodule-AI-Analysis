import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "/mnt/c/Users/aupat/Documents/Project's/Thyroide/Thyroide_Project/data/TN5000_NoduleCrop"

RESULTS_DIR = os.path.expanduser(
    "~/thyroid_ai/results_v3"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

FROZEN_EPOCHS = 8
FINETUNE_EPOCHS = 12

INITIAL_LR = 1e-4
FINETUNE_LR = 1e-5

SEED = 42

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# CPU CHECK
# ============================================================

print("=" * 70)
print("TN5000 V3 NODULE-CROP TRAINING")
print("=" * 70)

print(f"\nTensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices("GPU")
cpus = tf.config.list_physical_devices("CPU")

print("GPU devices:", gpus)
print("CPU devices:", len(cpus))

if gpus:
    print("\nWARNING: GPU detected.")
    print("This project is intended to run CPU-only.")

else:
    print("\nCPU-only training confirmed.")


# ============================================================
# DATA DIRECTORIES
# ============================================================

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "train"
)

VAL_DIR = os.path.join(
    DATA_DIR,
    "val"
)

TEST_DIR = os.path.join(
    DATA_DIR,
    "test"
)

print("\nDataset:")
print(DATA_DIR)

print("\nTrain:")
print(TRAIN_DIR)

print("Validation:")
print(VAL_DIR)

print("Test:")
print(TEST_DIR)


# ============================================================
# DATA GENERATORS
# ============================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.12,
    horizontal_flip=True,

    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=True,
    seed=SEED
)


val_gen = val_datagen.flow_from_directory(
    VAL_DIR,

    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False
)


print("\nClasses:")
print(train_gen.class_indices)

print(
    f"\nTrain images: {train_gen.samples}"
)

print(
    f"Validation images: {val_gen.samples}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = train_gen.classes

class_values = np.unique(classes)

weights = compute_class_weight(
    class_weight="balanced",
    classes=class_values,
    y=classes
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        class_values,
        weights
    )
}

print("\nClass weights:")
print(class_weights)


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING EFFICIENTNETB0")
print("=" * 70)

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

base_model.trainable = False


inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=regularizers.l2(0.01)
)(x)

x = layers.Dropout(0.50)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE FROZEN STAGE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LR
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


print("\nModel created.")

model.summary()


# ============================================================
# CALLBACKS - STAGE 1
# ============================================================

best_frozen_path = os.path.join(
    RESULTS_DIR,
    "best_frozen_model.keras"
)

frozen_checkpoint = ModelCheckpoint(
    best_frozen_path,

    monitor="val_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)

frozen_early_stop = EarlyStopping(
    monitor="val_auc",

    mode="max",

    patience=3,

    restore_best_weights=True,

    verbose=1
)

frozen_reduce_lr = ReduceLROnPlateau(
    monitor="val_auc",

    mode="max",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# STAGE 1 TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1: FROZEN BASE")
print("=" * 70)

history_frozen = model.fit(
    train_gen,

    validation_data=val_gen,

    epochs=FROZEN_EPOCHS,

    class_weight=class_weights,

    callbacks=[
        frozen_checkpoint,
        frozen_early_stop,
        frozen_reduce_lr
    ],

    verbose=1
)


# ============================================================
# FINE-TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2: FINE-TUNING")
print("=" * 70)

base_model.trainable = True


# Open only the last 80 EfficientNet layers.
# Keep BatchNormalization frozen for stability.

for layer in base_model.layers[:-80]:
    layer.trainable = False

for layer in base_model.layers[-80:]:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False

    else:
        layer.trainable = True


trainable_count = sum(
    1
    for layer in model.layers
    if layer.trainable
)

print(
    f"\nTrainable layers: {trainable_count}"
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FINETUNE_LR
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


# ============================================================
# CALLBACKS - STAGE 2
# ============================================================

best_finetuned_path = os.path.join(
    RESULTS_DIR,
    "best_finetuned_model.keras"
)

finetune_checkpoint = ModelCheckpoint(
    best_finetuned_path,

    monitor="val_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)

finetune_early_stop = EarlyStopping(
    monitor="val_auc",

    mode="max",

    patience=4,

    restore_best_weights=True,

    verbose=1
)

finetune_reduce_lr = ReduceLROnPlateau(
    monitor="val_auc",

    mode="max",

    factor=0.3,

    patience=2,

    min_lr=1e-8,

    verbose=1
)


# ============================================================
# STAGE 2 TRAINING
# ============================================================

history_finetune = model.fit(
    train_gen,

    validation_data=val_gen,

    epochs=FINETUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=[
        finetune_checkpoint,
        finetune_early_stop,
        finetune_reduce_lr
    ],

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    RESULTS_DIR,
    "TN5000_EfficientNetB0_V3_NoduleCrop_final.keras"
)

model.save(
    final_model_path
)

print("\nFinal model saved:")
print(final_model_path)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_rows = []


for stage_name, history in [
    ("frozen", history_frozen),
    ("fine_tune", history_finetune)
]:

    for epoch in range(
        len(history.history["loss"])
    ):

        row = {
            "stage": stage_name,
            "epoch": epoch + 1
        }

        for key, values in history.history.items():

            row[key] = values[epoch]

        history_rows.append(row)


history_df = pd.DataFrame(
    history_rows
)

history_path = os.path.join(
    RESULTS_DIR,
    "training_history_v3.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

print(
    f"\nTraining history saved:\n{history_path}"
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "tensorflow_version": tf.__version__,

    "model": "EfficientNetB0",

    "experiment": "V3 Nodule Crop",

    "dataset": "TN5000_NoduleCrop",

    "dataset_images": 4933,

    "train_images": train_gen.samples,

    "validation_images": val_gen.samples,

    "test_images": 1000,

    "image_size": list(IMG_SIZE),

    "batch_size": BATCH_SIZE,

    "frozen_epochs": FROZEN_EPOCHS,

    "finetune_epochs": FINETUNE_EPOCHS,

    "initial_learning_rate": INITIAL_LR,

    "finetune_learning_rate": FINETUNE_LR,

    "class_weights": class_weights,

    "augmentation": {
        "rotation_range": 12,
        "width_shift_range": 0.08,
        "height_shift_range": 0.08,
        "zoom_range": 0.12,
        "horizontal_flip": True
    },

    "nodule_crop_padding_ratio": 0.10,

    "threshold_selection": (
        "Validation only; test set not used during training."
    )
}

config_path = os.path.join(
    RESULTS_DIR,
    "training_configuration_v3.json"
)

with open(
    config_path,
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )

print(
    f"\nConfiguration saved:\n{config_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

best_val_auc = max(
    history_df["val_auc"]
)

print("\n" + "=" * 70)
print("V3 TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest validation AUC: {best_val_auc:.6f}"
)

print("\nTest set was NOT used.")

print("\nResults directory:")
print(RESULTS_DIR)

print("\nNext step:")
print("Evaluate V3 validation predictions and select threshold.")
