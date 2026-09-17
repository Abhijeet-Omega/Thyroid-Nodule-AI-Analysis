# ================== IMPORTS ==================
import os, csv
import numpy as np
import tensorflow as tf
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

# ================== PARAMETERS ==================
IMG_SIZE = (224, 224)
MODEL_PATH = "thyroid_final_model.h5"
THRESHOLD = 0.45   # tuned threshold (better than 0.5)

# ================== TRAIN MODEL ==================
def train_model():
    DATA_DIR = r"C:\Users\aupat\Documents\Project's\Thyroide\data\Updated_data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        zoom_range=0.25,
        shear_range=0.15,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE,
        batch_size=16, class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE,
        batch_size=16, class_mode='binary',
        subset='validation'
    )

    class_weights = compute_class_weight(
        'balanced', classes=np.unique(train_gen.classes), y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))

    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.6)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    cb = [
        callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_auc', mode='max', save_best_only=True),
        callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3)
    ]

    model.fit(train_gen, validation_data=val_gen,
              epochs=20, class_weight=class_weights, callbacks=cb)

    # Fine-tuning
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    model.fit(train_gen, validation_data=val_gen, epochs=10)

    model.save(MODEL_PATH)
    print("Model saved successfully!")

# ================== SAFE MODEL LOAD ==================
def load_model_safe():
    try:
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print("Model load error:", e)
    return None

model = load_model_safe()

# ================== PREDICTION ==================
def predict_image(path):
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize(IMG_SIZE)
        arr = np.array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        pred = model.predict(arr)[0][0]

        if pred > THRESHOLD:
            return "Malignant", float(pred)
        else:
            return "Benign", float(1 - pred)

    except Exception as e:
        print("Prediction error:", e)
        return "Error", 0.0

# ================== LOGGING ==================
def log_result(path, label, conf):
    try:
        file = "prediction_log.csv"
        write_header = not os.path.exists(file)

        with open(file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Time","Image","Prediction","Confidence"])
            writer.writerow([datetime.now(), path, label, conf])
    except:
        pass

# ================== GUI ==================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Thyroid Detection")
        self.root.geometry("650x700")
        self.root.configure(bg="#ECF0F1")

        self.img_path = None

        frame = tk.Frame(root, bg="white", padx=20, pady=20)
        frame.pack(padx=30, pady=30, fill="both", expand=True)

        tk.Label(frame, text="Thyroid Detection System",
                 font=("Segoe UI", 22, "bold"),
                 bg="white").pack(pady=10)

        self.canvas = tk.Canvas(frame, width=300, height=300, bg="#BDC3C7")
        self.canvas.pack(pady=15)

        self.result = tk.Label(frame, text="Prediction: -",
                              font=("Segoe UI", 18, "bold"),
                              bg="white")
        self.result.pack()

        self.conf = tk.Label(frame, text="Confidence: -",
                            font=("Segoe UI", 14),
                            bg="white")
        self.conf.pack()

        tk.Button(frame, text="Upload Image", command=self.load,
                  bg="#27AE60", fg="white").pack(pady=10)

        tk.Button(frame, text="Predict", command=self.predict,
                  bg="#2980B9", fg="white").pack(pady=10)

    def load(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_path = path
            img = Image.open(path)
            img = ImageOps.fit(img, (300,300))
            self.tkimg = ImageTk.PhotoImage(img)
            self.canvas.create_image(150,150,image=self.tkimg)

    def predict(self):
        if not self.img_path:
            messagebox.showwarning("Warning","Upload image first")
            return

        if model is None:
            messagebox.showerror("Error","Model not loaded")
            return

        label, conf = predict_image(self.img_path)

        color = "green" if label=="Benign" else "red"
        self.result.config(text=f"Prediction: {label}", fg=color)
        self.conf.config(text=f"Confidence: {conf:.2%}")

        log_result(self.img_path, label, conf)

# ================== RUN ==================
def run_gui():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    print("1 Train Model\n2 Run GUI")
    ch = input("Enter choice: ")

    if ch == "1":
        train_model()
    else:
        run_gui()
