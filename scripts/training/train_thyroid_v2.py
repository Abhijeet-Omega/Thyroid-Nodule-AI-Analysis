import os

# ============================================================
# FORCE CPU
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import random
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
    ReduceLROnPlateau,
    CSVLogger
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = (
    "/mnt/c/Users/aupat/Documents/Project's/"
    "Thyroide/Thyroide_Project/data/TN5000_LeakageSafe"
)

RESULTS_DIR = os.path.expanduser(
    "~/thyroid_ai/results_v2"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

FROZEN_EPOCHS = 8
FINETUNE_EPOCHS = 12

INITIAL_LR = 1e-4
FINETUNE_LR = 1e-5

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# CPU CHECK
# ============================================================

print("=" * 70)
print("TN5000 EFFICIENTNETB0 — MODEL V2")
print("=" * 70)

print("\nTensorFlow version:", tf.__version__)

print(
    "GPU devices:",
    tf.config.list_physical_devices("GPU")
)

print(
    "CPU devices:",
    tf.config.list_physical_devices("CPU")
)

# ============================================================
# DATA DIRECTORIES
# ============================================================

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

print("\nDataset:")
print(DATA_DIR)

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

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nClass mapping:")
print(train_generator.class_indices)

print(
    "\nTraining images:",
    train_generator.samples
)

print(
    "Validation images:",
    val_generator.samples
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = train_generator.classes

class_values = np.unique(classes)

weights = compute_class_weight(
    class_weight="balanced",
    classes=class_values,
    y=classes
)

class_weights = {
    int(class_values[i]): float(weights[i])
    for i in range(len(class_values))
}

print("\nClass weights:")
print(class_weights)

# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MODEL")
print("=" * 70)

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = layers.Input(
    shape=(224, 224, 3)
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
    kernel_regularizer=regularizers.l2(0.005)
)(x)

x = layers.Dropout(0.45)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(
    inputs,
    outputs
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LR
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.AUC(
            name="auc"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)

model.summary()

# ============================================================
# CALLBACKS — FROZEN STAGE
# ============================================================

frozen_best_path = os.path.join(
    RESULTS_DIR,
    "best_frozen_model.keras"
)

frozen_history_path = os.path.join(
    RESULTS_DIR,
    "frozen_training_history.csv"
)

callbacks_frozen = [

    ModelCheckpoint(
        frozen_best_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    CSVLogger(
        frozen_history_path
    )
]

# ============================================================
# STAGE 1 — FROZEN BASE
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 — FROZEN EFFICIENTNETB0")
print("=" * 70)

history_frozen = model.fit(
    train_generator,

    validation_data=val_generator,

    epochs=FROZEN_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_frozen,

    verbose=1
)

# ============================================================
# LOAD BEST FROZEN MODEL
# ============================================================

if os.path.exists(frozen_best_path):

    model = tf.keras.models.load_model(
        frozen_best_path
    )

    print(
        "\nLoaded best frozen model."
    )

# ============================================================
# STAGE 2 — FINE-TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 — FINE-TUNING")
print("=" * 70)

base_model = model.layers[1]

base_model.trainable = True

# Freeze everything except the last 80 layers.
for layer in base_model.layers[:-80]:

    layer.trainable = False

# Keep BatchNormalization layers frozen.
for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False

trainable_count = sum(
    1
    for layer in base_model.layers
    if layer.trainable
)

print(
    "\nTrainable EfficientNet layers:",
    trainable_count
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
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)

finetuned_best_path = os.path.join(
    RESULTS_DIR,
    "best_finetuned_model.keras"
)

finetuned_history_path = os.path.join(
    RESULTS_DIR,
    "finetuning_history.csv"
)

callbacks_finetune = [

    ModelCheckpoint(
        finetuned_best_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    CSVLogger(
        finetuned_history_path
    )
]

history_finetune = model.fit(
    train_generator,

    validation_data=val_generator,

    epochs=FINETUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_finetune,

    verbose=1
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    RESULTS_DIR,
    "TN5000_EfficientNetB0_V2_final.keras"
)

model.save(
    final_model_path
)

print(
    "\nFinal V2 model saved:"
)

print(final_model_path)

# ============================================================
# COMBINE TRAINING HISTORY
# ============================================================

frozen_df = pd.DataFrame(
    history_frozen.history
)

finetune_df = pd.DataFrame(
    history_finetune.history
)

frozen_df["stage"] = "frozen"
finetune_df["stage"] = "fine_tuning"

combined_history = pd.concat(
    [
        frozen_df,
        finetune_df
    ],
    ignore_index=True
)

combined_history.insert(
    0,
    "epoch",
    range(
        1,
        len(combined_history) + 1
    )
)

history_path = os.path.join(
    RESULTS_DIR,
    "training_history.csv"
)

combined_history.to_csv(
    history_path,
    index=False
)

# ============================================================
# CONFIGURATION FILE
# ============================================================

configuration = {

    "model": "EfficientNetB0",

    "version": "V2",

    "dataset": "TN5000_LeakageSafe",

    "train_images": int(
        train_generator.samples
    ),

    "validation_images": int(
        val_generator.samples
    ),

    "test_images": 1000,

    "image_size": [
        224,
        224
    ],

    "batch_size": BATCH_SIZE,

    "frozen_epochs_requested":
        FROZEN_EPOCHS,

    "finetune_epochs_requested":
        FINETUNE_EPOCHS,

    "initial_learning_rate":
        INITIAL_LR,

    "finetune_learning_rate":
        FINETUNE_LR,

    "fine_tune_last_layers":
        80,

    "batch_normalization_frozen":
        True,

    "augmentation": {

        "rotation_range": 12,

        "width_shift_range": 0.08,

        "height_shift_range": 0.08,

        "zoom_range": 0.12,

        "contrast_range": [
            0.85,
            1.15
        ],

        "horizontal_flip": True
    },

    "class_mapping":
        train_generator.class_indices,

    "class_weights":
        class_weights,

    "test_used_during_training":
        False,

    "random_seed":
        SEED
}

config_path = os.path.join(
    RESULTS_DIR,
    "training_configuration.json"
)

with open(
    config_path,
    "w"
) as f:

    json.dump(
        configuration,
        f,
        indent=4
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V2 TRAINING COMPLETE")
print("=" * 70)

print("\nResults directory:")
print(RESULTS_DIR)

print("\nSaved files:")

for filename in sorted(
    os.listdir(RESULTS_DIR)
):

    print(
        " -",
        filename
    )

print("\nIMPORTANT:")
print(
    "The test set was NOT used during training."
)

print(
    "Use the validation set to select the final threshold."
)

print("=" * 70)
