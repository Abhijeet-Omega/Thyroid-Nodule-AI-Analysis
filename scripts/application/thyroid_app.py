"""
================================================================================
THYROID NODULE AI ANALYSIS APPLICATION
TN5000 - EfficientNetB0 - V3 (Nodule Crop)
================================================================================

Academic decision-support demonstration only.
This system is not intended for clinical diagnosis, treatment, or replacement
of qualified medical professionals.

This application performs INFERENCE ONLY using an already trained and
validated TensorFlow/Keras model. It does not retrain, redesign, or otherwise
modify the model, the threshold, the preprocessing pipeline, or the dataset.

Expected location in the project:
    scripts/application/thyroid_app.py

Expected model location (resolved relative to this file, project_root/..):
    project_root / models / V3 / TN5000_EfficientNetB0_V3_NoduleCrop_final.keras

Run (from an activated .venv):
    python scripts/application/thyroid_app.py
================================================================================
"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import sys
import csv
import queue
import threading
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# TensorFlow / Keras (imported lazily-ish but at module load, per spec: load once)
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


# ==============================================================================
# 2. CONFIGURATION  (DO NOT MODIFY THESE VALUES)
# ==============================================================================

# Validated classification threshold selected during V3 validation.
# DO NOT CHANGE. DO NOT RE-OPTIMIZE.
DECISION_THRESHOLD = 0.55

# Exact padding ratio used when the V3 training crop dataset was generated
# from the TN5000 XML annotations. The application MUST replicate this
# exact ratio for user-selected ROIs.
PADDING_RATIO = 0.10

# Model input size expected by EfficientNetB0 V3.
MODEL_INPUT_SIZE = (224, 224)  # (width, height)

# Supported input image formats.
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Fallback size (in pixels) used for the ultrasound viewer ONLY before the
# canvas has been realized by the window manager (its very first layout
# pass). In normal operation the viewer is sized dynamically from the
# actual space Tkinter allocates to it (see _render_base_image), so the
# image always fits the window without clipping the sections below it.
# The image is only ever scaled for DISPLAY; all ROI math is converted
# back to original image coordinates before cropping.
MAX_DISPLAY_WIDTH = 480
MAX_DISPLAY_HEIGHT = 300

# Nodule crop preview box (display only, does not affect the 224x224 tensor
# that is actually sent to the model). This is a FIXED box so the lower
# center row has a guaranteed, predictable height that can never be pushed
# out of the window — the crop is fit completely inside it (never cropped,
# never distorted, never stretched) regardless of its aspect ratio.
PREVIEW_DISPLAY_SIZE = (272, 168)

# Project root resolution: this file lives at scripts/application/thyroid_app.py
# so project root is two levels up.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "V3" / "TN5000_EfficientNetB0_V3_NoduleCrop_final.keras"

# Optional prediction log location (created only if user enables logging).
LOG_DIR = PROJECT_ROOT / "results" / "Application"
LOG_FILE = LOG_DIR / "prediction_log.csv"

# Reference validated performance metrics (V3 held-out test set).
# These are DISPLAYED AS REFERENCE ONLY. They are NEVER recomputed by this
# application and are NEVER presented as the accuracy of any single uploaded
# image.
VALIDATED_METRICS = {
    "ROC-AUC": 0.894838,
    "Accuracy": 0.762000,
    "Precision": 0.940966,
    "Sensitivity": 0.719562,
    "Specificity": 0.877323,
    "F1": 0.815504,
    "Balanced Accuracy": 0.798443,
}


# ==============================================================================
# 3. MODEL LOADING (loaded once at application startup)
# ==============================================================================

class ModelHandle:
    """Holds the single loaded Keras model instance for the app's lifetime."""

    def __init__(self):
        self.model = None
        self.load_error = None
        self._load()

    def _load(self):
        if not MODEL_PATH.exists():
            self.load_error = (
                f"Model file not found.\n\nExpected at:\n{MODEL_PATH}\n\n"
                "Verify the project structure and that the V3 model has not "
                "been moved or renamed."
            )
            return
        try:
            # Loaded exactly once. No retraining, no recompilation changes.
            self.model = tf.keras.models.load_model(str(MODEL_PATH))
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"Failed to load model:\n{exc}"

    @property
    def is_ready(self):
        return self.model is not None


# ==============================================================================
# 4. IMAGE / ROI UTILITIES
# ==============================================================================

def load_image_rgb(path: str) -> Image.Image:
    """Load an image from disk, convert to RGB. Original file is never modified."""
    img = Image.open(path)
    img = img.convert("RGB")
    return img


def compute_display_size(orig_w: int, orig_h: int, max_w: int, max_h: int):
    """Compute a scaled-down display size that fits within (max_w, max_h),
    preserving aspect ratio. Never upscales beyond the original size."""
    scale = min(max_w / orig_w, max_h / orig_h, 1.0)
    disp_w = max(1, int(round(orig_w * scale)))
    disp_h = max(1, int(round(orig_h * scale)))
    return disp_w, disp_h


def fit_image_to_box(image: Image.Image, max_w: int, max_h: int,
                      allow_upscale: bool = False, max_upscale: float = 4.0):
    """Resize `image` so it fits COMPLETELY inside a (max_w x max_h) box,
    preserving its aspect ratio exactly. Never crops. Never distorts
    (stretches) the image. This is a pure DISPLAY concern — it has no
    effect on the model input pipeline, which always uses the original
    padded crop resized independently to 224x224.

    If allow_upscale is False (used for the main ultrasound viewer), the
    image is only ever scaled DOWN, matching the original viewer behavior.
    If True (used for the nodule crop preview, since crops can be very
    small), the image may be scaled up, bounded by max_upscale, so small
    crops remain legible instead of appearing as a tiny speck.

    Returns (resized_image, disp_w, disp_h).
    """
    orig_w, orig_h = image.size
    if orig_w <= 0 or orig_h <= 0 or max_w <= 0 or max_h <= 0:
        return image, orig_w, orig_h

    scale = min(max_w / orig_w, max_h / orig_h)
    if allow_upscale:
        scale = min(scale, max_upscale)
    else:
        scale = min(scale, 1.0)

    disp_w = max(1, int(round(orig_w * scale)))
    disp_h = max(1, int(round(orig_h * scale)))
    resized = image.resize((disp_w, disp_h), resample=Image.BILINEAR)
    return resized, disp_w, disp_h


def display_to_original_coords(x_disp, y_disp, orig_w, orig_h, disp_w, disp_h):
    """Convert display coordinates to deterministic integer image coordinates."""
    if disp_w <= 0 or disp_h <= 0 or orig_w <= 0 or orig_h <= 0:
        raise ValueError("Invalid image/display dimensions.")

    x_disp = min(max(float(x_disp), 0.0), float(disp_w))
    y_disp = min(max(float(y_disp), 0.0), float(disp_h))

    scale_x = orig_w / disp_w
    scale_y = orig_h / disp_h

    # Floor the upper-left and use ceil for the lower-right so a selected
    # display ROI does not unintentionally shrink during coordinate mapping.
    x_orig = int(np.floor(x_disp * scale_x))
    y_orig = int(np.floor(y_disp * scale_y))
    return x_orig, y_orig


def clamp_box(xmin, ymin, xmax, ymax, image_width, image_height):
    """Clamp a bounding box to valid image bounds, matching the exact
    clamping logic used when the V3 training crop dataset was built."""
    xmin = max(0, min(xmin, image_width - 1))
    ymin = max(0, min(ymin, image_height - 1))
    xmax = max(1, min(xmax, image_width))
    ymax = max(1, min(ymax, image_height))
    return xmin, ymin, xmax, ymax


def apply_training_padding(xmin, ymin, xmax, ymax, image_width, image_height):
    """Apply the EXACT 10% padding logic used during V3 training-crop
    generation from TN5000 XML annotations. Do not alter this ratio or
    this order of operations.
    """
    # Step 1: clamp raw (user-drawn) box to image bounds first.
    xmin, ymin, xmax, ymax = clamp_box(xmin, ymin, xmax, ymax, image_width, image_height)

    # Step 2: compute box dimensions.
    box_width = xmax - xmin
    box_height = ymax - ymin

    # Step 3: compute padding amounts (10% of box dimensions, integer pixels).
    pad_x = int(box_width * PADDING_RATIO)
    pad_y = int(box_height * PADDING_RATIO)

    # Step 4: apply padding.
    xmin_p = max(0, xmin - pad_x)
    ymin_p = max(0, ymin - pad_y)
    xmax_p = min(image_width, xmax + pad_x)
    ymax_p = min(image_height, ymax + pad_y)

    return int(xmin_p), int(ymin_p), int(xmax_p), int(ymax_p)


def crop_with_padding(image_rgb: Image.Image, roi_box_original):
    """Given the ORIGINAL-resolution ROI box (xmin, ymin, xmax, ymax) as
    drawn by the user, apply the training-identical 10% padding and return
    the cropped PIL image plus the final padded box.
    """
    orig_w, orig_h = image_rgb.size
    xmin, ymin, xmax, ymax = roi_box_original

    if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
        raise ValueError("Selected ROI has zero or negative width/height.")

    px1, py1, px2, py2 = apply_training_padding(xmin, ymin, xmax, ymax, orig_w, orig_h)

    if (px2 - px1) <= 0 or (py2 - py1) <= 0:
        raise ValueError("Padded ROI collapsed to zero size after clamping.")

    cropped = image_rgb.crop((px1, py1, px2, py2))
    return cropped, (px1, py1, px2, py2)


# ==============================================================================
# 5. PREPROCESSING + PREDICTION  (mirrors V3 training preprocessing exactly)
# ==============================================================================

def predict_nodule(crop_image: Image.Image, model_handle: ModelHandle) -> dict:
    """
    Run V3 inference on a nodule crop.

    Pipeline (must match training exactly):
        PIL crop -> RGB -> resize 224x224 -> numpy array
        -> EfficientNet preprocess_input -> model.predict()

    Returns a structured dict:
        {
            "probability": float,   # raw malignant probability from the model
            "prediction": "Malignant" | "Benign",
            "confidence": float,    # probability if Malignant, else 1 - probability
            "threshold": 0.55
        }
    """
    if not model_handle.is_ready:
        raise RuntimeError("Model is not loaded. Cannot run prediction.")

    # 1. Ensure RGB (training crops were always converted to RGB before cropping).
    crop_rgb = crop_image.convert("RGB")

    # 2. Resize to model input size. No other resizing strategy is used.
    resized = crop_rgb.resize(MODEL_INPUT_SIZE, resample=Image.BILINEAR)

    # 3. Convert to numpy array (float).
    arr = np.array(resized).astype(np.float32)

    # 4. Apply the EXACT EfficientNetB0 preprocess_input used at training time.
    #    Do not substitute a manual /255 or other normalization scheme.
    arr = preprocess_input(arr)

    # 5. Add batch dimension.
    batch = np.expand_dims(arr, axis=0)

    # 6. Run inference.
    raw_output = model_handle.model.predict(batch, verbose=0)

    # 7. Extract malignant probability (sigmoid, single unit output).
    probability = float(np.squeeze(raw_output))

    # 8. Apply the validated, fixed decision threshold. DO NOT RE-TUNE.
    prediction = "Malignant" if probability >= DECISION_THRESHOLD else "Benign"

    # 9. Confidence is reported relative to the predicted class, not as a
    #    clinical certainty score.
    confidence = probability if prediction == "Malignant" else (1.0 - probability)

    return {
        "probability": probability,
        "prediction": prediction,
        "confidence": confidence,
        "threshold": DECISION_THRESHOLD,
    }


# ==============================================================================
# 6. OPTIONAL PREDICTION LOGGING
# ==============================================================================

def log_prediction(record: dict):
    """Append a single prediction record to a CSV log. Never required for
    prediction to function; failures here must not crash the app and must
    not include patient-identifying information."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_exists = LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except Exception:  # noqa: BLE001
        # Logging is optional and must never interrupt the workflow.
        pass


# ==============================================================================
# 7. GUI
# ==============================================================================

BG_DARK = "#12181f"
BG_PANEL = "#1b232d"
BG_PANEL_ALT = "#202934"
FG_MAIN = "#e7edf3"
FG_MUTED = "#8fa1b3"
ACCENT = "#3aa0ff"
ACCENT_DIM = "#245a86"
GOOD = "#2fbf71"
BAD = "#e2543a"
BORDER = "#2c3742"


class ThyroidApp(tk.Tk):
    def __init__(self, model_handle: ModelHandle):
        super().__init__()
        self.model_handle = model_handle

        self.title("Thyroid Nodule AI Analysis — TN5000 / EfficientNetB0 / V3")
        self.geometry("1360x900")
        self.minsize(1180, 760)
        self.configure(bg=BG_DARK)

        # ---- state ----
        self.original_image: Image.Image | None = None
        self.original_path: str | None = None
        self.orig_w = 0
        self.orig_h = 0
        self.disp_w = 0
        self.disp_h = 0
        self.tk_display_image = None
        self.canvas_image_id = None

        self.roi_start = None          # (x, y) in canvas coords
        self.roi_rect_id = None
        self.roi_box_display = None    # (x1,y1,x2,y2) in canvas coords
        self.roi_box_original = None   # (x1,y1,x2,y2) in original image coords
        self.roi_confirmed = False

        # Offset (in canvas pixels) of the top-left corner of the fitted
        # ultrasound image within the canvas — the canvas fills its whole
        # cell, and the image is centered inside it, so mouse coordinates
        # must be translated by this offset before converting to original
        # image coordinates.
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self._last_canvas_size = None

        self.padded_crop_image: Image.Image | None = None
        self.padded_box_original = None
        self.tk_preview_image = None

        # Background inference plumbing: predict_nodule() runs on a worker
        # thread so the UI never freezes; results are handed back to the
        # Tkinter main thread through this queue and picked up via after().
        self._pred_queue = queue.Queue()
        self._pred_poll_job = None

        self.log_enabled = tk.BooleanVar(value=False)

        self._build_style()
        self._build_layout()

        if not self.model_handle.is_ready:
            self.after(200, lambda: messagebox.showerror(
                "Model Load Error",
                self.model_handle.load_error or "Unknown model loading error."
            ))

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG_PANEL)
        style.configure("Header.TFrame", background=BG_DARK)
        style.configure("TLabel", background=BG_PANEL, foreground=FG_MAIN,
                        font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG_PANEL, foreground=FG_MUTED,
                        font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG_DARK, foreground=FG_MAIN,
                        font=("Segoe UI", 19, "bold"))
        style.configure("Subtitle.TLabel", background=BG_DARK, foreground=FG_MUTED,
                        font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=BG_PANEL, foreground=ACCENT,
                        font=("Segoe UI", 11, "bold"))
        style.configure("CardTitle.TLabel", background=BG_PANEL_ALT, foreground=ACCENT,
                        font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"),
                        padding=(12, 9), foreground="#ffffff", background=ACCENT)
        style.map("Accent.TButton", background=[("active", "#1687ee")],
                  foreground=[("disabled", "#7f8c99")])
        style.configure("Footer.TLabel", background=BG_DARK, foreground=FG_MUTED,
                        font=("Segoe UI", 8))

    def _make_card(self, parent, bg=BG_PANEL, border=BORDER):
        return tk.Frame(parent, bg=bg, highlightbackground=border,
                        highlightthickness=1, bd=0)

    def _make_icon_label(self, parent, icon, title, bg=BG_PANEL):
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", padx=14, pady=(12, 7))
        tk.Label(row, text=icon, bg=bg, fg=ACCENT,
                 font=("Segoe UI Symbol", 15, "bold")).pack(side="left", padx=(0, 9))
        tk.Label(row, text=title, bg=bg, fg=ACCENT,
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        return row

    # ------------------------------------------------------------------
    # Layout — clean three-column dashboard inspired by the supplied
    # reference image. Only information/features already implemented in
    # this application are displayed.
    # ------------------------------------------------------------------
    def _build_layout(self):
        # =============================== HEADER =========================
        header = tk.Frame(self, bg=BG_DARK, height=106)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)

        title_row = tk.Frame(header, bg=BG_DARK)
        title_row.pack(fill="x", padx=24, pady=(14, 0))

        # Small decorative thyroid mark — no clinical data, just branding.
        tk.Label(title_row, text="♢", bg=BG_DARK, fg=ACCENT,
                 font=("Segoe UI Symbol", 30, "bold")).pack(side="left", padx=(0, 12))

        title_block = tk.Frame(title_row, bg=BG_DARK)
        title_block.pack(side="left", anchor="w")
        tk.Label(title_block, text="THYROID NODULE AI ANALYSIS",
                 bg=BG_DARK, fg=FG_MAIN,
                 font=("Segoe UI", 19, "bold")).pack(anchor="w")
        tk.Label(title_block, text="TN5000  •  EfficientNetB0  •  V3",
                 bg=BG_DARK, fg=FG_MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(1, 0))
        tk.Label(title_block, text="Ultrasound Image Analysis & AI-Based Classification",
                 bg=BG_DARK, fg=FG_MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(1, 0))

        # Decorative header accent (right side) — purely visual, no data claims.
        tagline_block = tk.Frame(header, bg=BG_DARK)
        tagline_block.pack(side="right", padx=24, pady=(24, 0))

        tagline_text = tk.Frame(tagline_block, bg=BG_DARK)
        tagline_text.pack(side="left", padx=(0, 12))
        tk.Label(tagline_text, text="Academic Research", bg=BG_DARK, fg=FG_MAIN,
                 font=("Segoe UI", 10, "bold"), justify="right").pack(anchor="e")
        tk.Label(tagline_text, text="AI Prototype System", bg=BG_DARK, fg=FG_MUTED,
                 font=("Segoe UI", 9), justify="right").pack(anchor="e")

        spark = tk.Canvas(tagline_block, width=90, height=34, bg=BG_DARK, highlightthickness=0)
        spark.pack(side="left")
        spark.create_line(0, 24, 14, 24, 22, 6, 30, 30, 38, 12, 46, 20, 54, 4,
                           62, 26, 70, 16, 90, 16, fill=ACCENT, width=2, smooth=True)

        # =============================== BODY ============================
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(side="top", fill="both", expand=True, padx=18, pady=(0, 8))

        body.columnconfigure(0, weight=25, minsize=330)
        body.columnconfigure(1, weight=47, minsize=500)
        body.columnconfigure(2, weight=28, minsize=360)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_center_panel(body)
        self._build_right_panel(body)

        # =============================== FOOTER =========================
        footer = tk.Frame(self, bg=BG_DARK, height=28)
        footer.pack(side="bottom", fill="x")
        footer.pack_propagate(False)
        tk.Label(footer, text="Thyroid Nodule AI Analysis  |  TN5000  |  EfficientNetB0  |  V3",
                 bg=BG_DARK, fg=FG_MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=20)
        tk.Label(footer, text="Built for Education  |  Academic Decision-Support Demonstration",
                 bg=BG_DARK, fg=FG_MUTED,
                 font=("Segoe UI", 8)).pack(side="right", padx=20)

    # ------------------------------------------------------------------
    # LEFT PANEL — input + instructions + ROI tools
    # ------------------------------------------------------------------
    def _build_left_panel(self, parent):
        panel = self._make_card(parent)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self._make_icon_label(panel, "⇧", "INPUT ULTRASOUND")

        btn_row = tk.Frame(panel, bg=BG_PANEL)
        btn_row.pack(fill="x", padx=18, pady=(0, 6))

        ttk.Button(btn_row, text="▣  Upload Image",
                   command=self.on_upload_image).pack(side="left", fill="x", expand=True)
        ttk.Button(btn_row, text="◉  New Image",
                   command=self.on_new_image).pack(side="left", fill="x", expand=True, padx=(8, 0))

        tk.Label(panel,
                 text="Supported format: PNG, JPG  |  TN5000 ultrasound images",
                 bg=BG_PANEL, fg=FG_MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w", padx=18, pady=(2, 10))

        # Instructions card
        instructions = tk.Frame(panel, bg=BG_PANEL_ALT,
                                highlightbackground=BORDER, highlightthickness=1)
        instructions.pack(fill="x", padx=18, pady=(0, 12))

        tk.Label(instructions, text="●  Instructions", bg=BG_PANEL_ALT,
                 fg=ACCENT, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 8))

        steps = (
            "Upload an ultrasound image",
            "Draw around the nodule",
            "Confirm the selected region",
            "Click Analyze Nodule",
        )
        for i, step in enumerate(steps, 1):
            row = tk.Frame(instructions, bg=BG_PANEL_ALT)
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=str(i), bg="#26384a", fg=FG_MAIN,
                     font=("Segoe UI", 9, "bold"), width=2, pady=3).pack(side="left", padx=(0, 8))
            tk.Label(row, text=step, bg=BG_PANEL_ALT, fg=FG_MAIN,
                     font=("Segoe UI", 9)).pack(side="left")

        tk.Frame(instructions, bg=BG_PANEL_ALT, height=8).pack()

        # ROI tools card
        self._make_icon_label(panel, "□", "ROI TOOLS")
        roi_row = tk.Frame(panel, bg=BG_PANEL)
        roi_row.pack(fill="x", padx=18, pady=(0, 12))
        ttk.Button(roi_row, text="⌫  Clear ROI",
                   command=self.on_clear_roi).pack(side="left", fill="x", expand=True)
        ttk.Button(roi_row, text="✓  Confirm ROI", style="Accent.TButton",
                   command=self.on_confirm_roi).pack(side="left", fill="x", expand=True, padx=(8, 0))

        # Keep this area purely informational/decorative; no unsupported
        # clinical measurements or extra application features are introduced.
        note = tk.Frame(panel, bg=BG_PANEL)
        note.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        tk.Label(note, text="Select the nodule region manually.",
                 bg=BG_PANEL, fg=FG_MUTED,
                 font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(8, 0))
        tk.Frame(note, bg=ACCENT, height=1, width=90).pack(anchor="w", pady=(9, 0))

        # Decorative branding block (bottom of left panel) — purely visual,
        # mirrors the reference dashboard's sign-off styling. Introduces no
        # new data, claims, or functionality.
        brand = tk.Frame(note, bg=BG_PANEL)
        brand.pack(side="bottom", fill="x", pady=(10, 0))
        tk.Label(brand, text="⌬", bg=BG_PANEL, fg=ACCENT_DIM,
                 font=("Segoe UI Symbol", 26)).pack(anchor="w")
        tk.Label(brand, text="Engineering prototype for\nautomated nodule classification.",
                 bg=BG_PANEL, fg=FG_MUTED, font=("Segoe UI", 9, "italic"),
                 justify="left").pack(anchor="w", pady=(4, 0))
        tk.Frame(brand, bg=BORDER, height=1, width=90).pack(anchor="w", pady=(8, 0))

    # ------------------------------------------------------------------
    # CENTER PANEL — full ultrasound + cropped ROI + ROI information
    # ------------------------------------------------------------------
    def _build_center_panel(self, parent):
        panel = self._make_card(parent)
        panel.grid(row=0, column=1, sticky="nsew", padx=8)

        # Explicit row budget: every row EXCEPT the image viewer keeps its
        # natural (content-driven) height no matter what — only row 2 (the
        # ultrasound viewer) is allowed to grow or shrink to absorb
        # whatever space is left. This is what guarantees the lower cards
        # (crop preview + ROI information) are always fully visible.
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(0, weight=0)  # header
        panel.grid_rowconfigure(1, weight=0)  # instruction line
        panel.grid_rowconfigure(2, weight=1)  # ultrasound viewer (flexible)
        panel.grid_rowconfigure(3, weight=0)  # filename / dimensions
        panel.grid_rowconfigure(4, weight=0)  # crop preview + ROI info row

        header_row = tk.Frame(panel, bg=BG_PANEL)
        header_row.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 7))
        tk.Label(header_row, text="⌁", bg=BG_PANEL, fg=ACCENT,
                 font=("Segoe UI Symbol", 15, "bold")).pack(side="left", padx=(0, 9))
        tk.Label(header_row, text="ULTRASOUND IMAGE", bg=BG_PANEL, fg=ACCENT,
                 font=("Segoe UI", 11, "bold")).pack(side="left")

        instruction = tk.Label(
            panel,
            text="Draw a bounding box around the nodule  •  Confirm ROI  •  Click Analyze",
            bg=BG_PANEL, fg=FG_MUTED, font=("Segoe UI", 9)
        )
        instruction.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 7))

        # ---- Row 2: ultrasound viewer. The canvas fills this row entirely
        # and is re-fitted to whatever space it is actually given (see
        # _render_base_image / on_canvas_resize), so the image is always
        # shown as large as possible WITHOUT ever pushing the rows below
        # it out of the window.
        image_card = tk.Frame(panel, bg="#080c11",
                              highlightbackground=BORDER, highlightthickness=1)
        image_card.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 5))
        image_card.grid_rowconfigure(0, weight=1)
        image_card.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            image_card, bg="#080c11", highlightthickness=0, cursor="crosshair"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        meta_row = tk.Frame(panel, bg=BG_PANEL)
        meta_row.grid(row=3, column=0, sticky="ew", padx=14, pady=(2, 7))
        self.lbl_file_meta = tk.Label(meta_row, text="No image loaded",
                                      bg=BG_PANEL, fg=FG_MUTED,
                                      font=("Segoe UI", 8))
        self.lbl_file_meta.pack(side="left")

        # ---- Row 4: lower center area (crop preview + ROI information).
        # This row's height is fixed by its content (a fixed-size preview
        # box, see PREVIEW_DISPLAY_SIZE) and is never sacrificed to the
        # image viewer above it — it always keeps its full natural size.
        lower = tk.Frame(panel, bg=BG_PANEL)
        lower.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 12))
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=1)

        preview_card = tk.Frame(lower, bg=BG_PANEL_ALT,
                                highlightbackground=BORDER, highlightthickness=1)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(preview_card, text="◈  NODULE PREVIEW (CROPPED ROI)",
                 bg=BG_PANEL_ALT, fg=ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(9, 6))

        preview_box = tk.Frame(preview_card, bg="#080c11")
        preview_box.pack(padx=10, pady=(0, 10))
        # Fixed-size canvas: the padded crop (whatever its aspect ratio) is
        # always fit completely inside it via fit_image_to_box and centered
        # — never clipped, never stretched. See on_confirm_roi.
        self.preview_canvas = tk.Canvas(
            preview_box, width=PREVIEW_DISPLAY_SIZE[0], height=PREVIEW_DISPLAY_SIZE[1],
            bg="#080c11", highlightthickness=0
        )
        self.preview_canvas.pack()

        info_frame = tk.Frame(lower, bg=BG_PANEL_ALT,
                              highlightbackground=BORDER, highlightthickness=1)
        info_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(info_frame, text="●  ROI INFORMATION", bg=BG_PANEL_ALT,
                 fg=ACCENT, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(9, 6))

        self.lbl_orig_size = self._info_row(info_frame, "Original Image:")
        self.lbl_roi_coords = self._info_row(info_frame, "Selected ROI:")
        self.lbl_padded_coords = self._info_row(info_frame, "Padded ROI:")
        self.lbl_padding = self._info_row(info_frame, "Padding:")
        self.lbl_crop_size = self._info_row(info_frame, "Crop Size:")
        tk.Frame(info_frame, bg=BG_PANEL_ALT, height=8).pack()

    def _info_row(self, parent, label_text):
        row = tk.Frame(parent, bg=BG_PANEL_ALT)
        row.pack(fill="x", padx=10, pady=2)
        tk.Label(row, text=label_text, bg=BG_PANEL_ALT, fg=FG_MUTED,
                 font=("Segoe UI", 8), width=15, anchor="w").pack(side="left")
        value_lbl = tk.Label(row, text="—", bg=BG_PANEL_ALT, fg=FG_MAIN,
                             font=("Segoe UI", 9, "bold"), anchor="w")
        value_lbl.pack(side="left", fill="x", expand=True)
        return value_lbl

    # ------------------------------------------------------------------
    # RIGHT PANEL — prediction + validated reference metrics + note
    # ------------------------------------------------------------------
    def _build_right_panel(self, parent):
        panel = self._make_card(parent)
        panel.grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        self._make_icon_label(panel, "◉", "AI PREDICTION")

        badge_row = tk.Frame(panel, bg=BG_PANEL)
        badge_row.pack(fill="x", padx=14, pady=(0, 7))
        tk.Label(badge_row, text="V3 MODEL", bg=BG_PANEL_ALT, fg=ACCENT,
                 font=("Segoe UI", 8, "bold"), padx=10, pady=4,
                 highlightbackground=ACCENT_DIM, highlightthickness=1).pack(side="right")

        self.result_banner = tk.Frame(panel, bg=BG_PANEL_ALT)
        self.result_banner.pack(fill="x", padx=14, pady=(0, 10))
        self._result_inner = tk.Frame(self.result_banner, bg=BG_PANEL_ALT, pady=12)
        self._result_inner.pack(fill="x")
        self.result_icon = tk.Label(
            self._result_inner, text="●", font=("Segoe UI", 18),
            bg=BG_PANEL_ALT, fg=FG_MUTED
        )
        self.result_icon.pack(side="left", padx=(16, 8))
        self.result_text = tk.Label(
            self._result_inner, text="NO PREDICTION YET",
            font=("Segoe UI", 18, "bold"),
            bg=BG_PANEL_ALT, fg=FG_MUTED
        )
        self.result_text.pack(side="left")

        metrics_frame = tk.Frame(panel, bg=BG_PANEL)
        metrics_frame.pack(fill="x", padx=14)

        self.lbl_prob = self._metric_row(metrics_frame, "Malignant Probability:")
        self.lbl_conf = self._metric_row(metrics_frame, "Model Confidence:")
        self.lbl_thresh = self._metric_row(metrics_frame, "Decision Threshold:", value="0.55")
        self.lbl_model = self._metric_row(metrics_frame, "Model:", value="TN5000 V3 — EfficientNetB0")
        self._metric_row(metrics_frame, "Input:", value="Padded ROI → 224×224")
        self._metric_row(metrics_frame, "Preprocess:", value="EfficientNetB0 preprocess_input")

        self.lbl_status = tk.Label(
            panel, text="Ready — upload an ultrasound image.",
            bg=BG_PANEL, fg=FG_MUTED,
            font=("Segoe UI", 8), anchor="w", justify="left", wraplength=330
        )
        self.lbl_status.pack(fill="x", padx=14, pady=(5, 2))

        tk.Label(
            panel,
            text="Confidence reflects the model's own output probability, not clinical certainty.",
            bg=BG_PANEL, fg=FG_MUTED, font=("Segoe UI", 8),
            anchor="w", justify="left", wraplength=330
        ).pack(fill="x", padx=14, pady=(1, 6))

        # Analyze action — this is the existing application function.
        self.btn_analyze = ttk.Button(
            panel, text="▶  Analyze Nodule", style="Accent.TButton",
            command=self.on_analyze
        )
        self.btn_analyze.pack(fill="x", padx=14, pady=(0, 6))

        self.log_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            panel, text="Log this prediction (no patient-identifying data)",
            variable=self.log_enabled
        ).pack(anchor="w", padx=14, pady=(0, 6))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=14, pady=4)

        # Reference metrics: existing validated values only.
        head = tk.Frame(panel, bg=BG_PANEL)
        head.pack(fill="x", padx=14, pady=(1, 3))
        tk.Label(head, text="▥  REFERENCE V3 PERFORMANCE", bg=BG_PANEL,
                 fg=ACCENT, font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Label(head, text="Reference Values", bg=BG_PANEL_ALT, fg=FG_MUTED,
                 font=("Segoe UI", 7, "bold"), padx=7, pady=3).pack(side="right")

        tk.Label(panel, text="Fixed held-out test values • reference only",
                 bg=BG_PANEL, fg=FG_MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(0, 4))

        ref_frame = tk.Frame(panel, bg=BG_PANEL_ALT,
                             highlightbackground=BORDER, highlightthickness=1)
        ref_frame.pack(fill="x", padx=14, pady=(0, 6))

        metric_pairs = [
            ("ROC-AUC", VALIDATED_METRICS["ROC-AUC"], "Accuracy", VALIDATED_METRICS["Accuracy"]),
            ("Precision", VALIDATED_METRICS["Precision"], "Sensitivity", VALIDATED_METRICS["Sensitivity"]),
            ("Specificity", VALIDATED_METRICS["Specificity"], "F1 Score", VALIDATED_METRICS["F1"]),
        ]
        for row_idx, pair in enumerate(metric_pairs):
            for col_idx in range(2):
                name, val = pair[col_idx * 2], pair[col_idx * 2 + 1]
                cell = tk.Frame(ref_frame, bg=BG_PANEL_ALT)
                cell.grid(row=row_idx, column=col_idx, sticky="ew", padx=9, pady=2)
                ref_frame.columnconfigure(col_idx, weight=1)
                tk.Label(cell, text=name, bg=BG_PANEL_ALT, fg=FG_MUTED,
                         font=("Segoe UI", 8)).pack(side="left")
                tk.Label(cell, text=f"{val:.4f}", bg=BG_PANEL_ALT, fg=FG_MAIN,
                         font=("Segoe UI", 9, "bold")).pack(side="right")

        # Academic-use note contains only the application's existing disclaimer.
        note = tk.Frame(panel, bg=BG_PANEL_ALT,
                        highlightbackground=BORDER, highlightthickness=1)
        note.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(note, text="●  ACADEMIC USE ONLY", bg=BG_PANEL_ALT, fg=ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 3))
        tk.Label(
            note,
            text="This tool is for academic research and educational purposes only.\n"
                 "Not intended for clinical diagnosis, treatment, or replacement of qualified medical professionals.",
            bg=BG_PANEL_ALT, fg=FG_MUTED,
            font=("Segoe UI", 8), justify="left", anchor="w", wraplength=330
        ).pack(fill="x", padx=10, pady=(0, 7))

    def _metric_row(self, parent, label_text, value="—", muted=False):
        row = tk.Frame(parent, bg=BG_PANEL)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label_text, bg=BG_PANEL, fg=FG_MUTED,
                 font=("Segoe UI", 8), anchor="w").pack(side="left")
        color = FG_MUTED if muted else FG_MAIN
        weight = "normal" if muted else "bold"
        value_lbl = tk.Label(row, text=value, bg=BG_PANEL, fg=color,
                             font=("Segoe UI", 9, weight), anchor="e")
        value_lbl.pack(side="right", fill="x", expand=True)
        return value_lbl

    # ==================================================================
    # EVENT HANDLERS
    # ==================================================================

    def on_upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Ultrasound Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            messagebox.showerror("Unsupported File", f"Unsupported file type: {ext}")
            return

        try:
            img = load_image_rgb(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Image Load Error", f"Could not open image:\n{exc}")
            return

        self._reset_state()
        self.original_image = img
        self.original_path = path
        self.orig_w, self.orig_h = img.size

        self.canvas.delete("all")
        self._render_base_image()

        self.lbl_orig_size.config(text=f"{self.orig_w} × {self.orig_h}")
        self.lbl_file_meta.config(text=f"{Path(path).name}  |  {self.orig_w} × {self.orig_h}")
        self.lbl_status.config(text="Image loaded — draw tightly around the nodule.")

    # ---- Responsive ultrasound viewer rendering ----
    def on_canvas_resize(self, event):
        """The ultrasound viewer's canvas fills whatever space the grid
        layout allocates to it. When that space changes (e.g. the window
        is resized), re-fit and re-center the displayed image so it is
        always shown as large as possible without ever being clipped or
        distorted. Guarded so it only does work when the size actually
        changed, to keep ROI dragging smooth."""
        new_size = (event.width, event.height)
        if self._last_canvas_size == new_size:
            return
        self._last_canvas_size = new_size
        if self.original_image is not None:
            self._render_base_image()

    def _render_base_image(self):
        """Fit the currently loaded ultrasound image into the canvas's
        actual current size, center it, and redraw it. Never crops, never
        distorts, never upscales beyond the original resolution."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT

        display_img, disp_w, disp_h = fit_image_to_box(
            self.original_image, canvas_w, canvas_h, allow_upscale=False
        )
        self.disp_w, self.disp_h = disp_w, disp_h
        self.canvas_offset_x = max(0, (canvas_w - disp_w) // 2)
        self.canvas_offset_y = max(0, (canvas_h - disp_h) // 2)

        self.tk_display_image = ImageTk.PhotoImage(display_img)
        self.canvas.delete("base_image")
        self.canvas.create_image(
            self.canvas_offset_x, self.canvas_offset_y, anchor="nw",
            image=self.tk_display_image, tags="base_image"
        )
        self.canvas.tag_lower("base_image")

        self._redraw_confirmed_roi_overlay()

    def _redraw_confirmed_roi_overlay(self):
        """Redraw the confirmed padded-ROI rectangle at the current image
        scale/offset. Any unconfirmed (in-progress) selection is tied to
        the previous canvas size and is cleared, since it can no longer be
        represented correctly after a resize."""
        if self.roi_rect_id is not None:
            self.canvas.delete(self.roi_rect_id)
            self.roi_rect_id = None
        self.roi_box_display = None

        if self.roi_confirmed and self.padded_box_original and self.orig_w and self.orig_h:
            px1, py1, px2, py2 = self.padded_box_original
            dx1 = self.canvas_offset_x + px1 * self.disp_w / self.orig_w
            dy1 = self.canvas_offset_y + py1 * self.disp_h / self.orig_h
            dx2 = self.canvas_offset_x + px2 * self.disp_w / self.orig_w
            dy2 = self.canvas_offset_y + py2 * self.disp_h / self.orig_h
            self.roi_rect_id = self.canvas.create_rectangle(
                dx1, dy1, dx2, dy2, outline=GOOD, width=3
            )

    def on_new_image(self):
        self._reset_state()
        self.canvas.delete("all")
        self.preview_canvas.delete("all")
        self._reset_prediction_ui()
        self.lbl_orig_size.config(text="—")
        self.lbl_roi_coords.config(text="—")
        self.lbl_padded_coords.config(text="—")
        self.lbl_padding.config(text="—")
        self.lbl_crop_size.config(text="—")
        self.lbl_file_meta.config(text="No image loaded")
        self.lbl_status.config(text="Ready — upload an ultrasound image.")

    def _reset_state(self):
        self.original_image = None
        self.original_path = None
        self.orig_w = self.orig_h = 0
        self.disp_w = self.disp_h = 0
        self.canvas_offset_x = self.canvas_offset_y = 0
        self.roi_start = None
        self.roi_rect_id = None
        self.roi_box_display = None
        self.roi_box_original = None
        self.roi_confirmed = False
        self.padded_crop_image = None
        self.padded_box_original = None

    def _reset_prediction_ui(self):
        self.result_banner.config(bg=BG_PANEL_ALT)
        self._result_inner.config(bg=BG_PANEL_ALT)
        self.result_icon.config(text="●", bg=BG_PANEL_ALT, fg=FG_MUTED)
        self.result_text.config(text="NO PREDICTION YET", bg=BG_PANEL_ALT, fg=FG_MUTED)
        self.lbl_prob.config(text="—")
        self.lbl_conf.config(text="—")
        if hasattr(self, "lbl_status") and self.original_image is None:
            self.lbl_status.config(text="Ready — upload an ultrasound image.")

    # ---- ROI drawing ----
    # Mouse coordinates are clamped to the region the image actually
    # occupies within the canvas (canvas_offset_x/y .. +disp_w/disp_h),
    # since the canvas fills its whole grid cell but the image is only
    # drawn centered within it.
    def _clamp_to_image_area(self, x, y):
        x_min, x_max = self.canvas_offset_x, self.canvas_offset_x + self.disp_w
        y_min, y_max = self.canvas_offset_y, self.canvas_offset_y + self.disp_h
        return min(max(x, x_min), x_max), min(max(y, y_min), y_max)

    def on_mouse_down(self, event):
        if self.original_image is None:
            return
        self.roi_confirmed = False
        start_x, start_y = self._clamp_to_image_area(event.x, event.y)
        self.roi_start = (start_x, start_y)
        if self.roi_rect_id is not None:
            self.canvas.delete(self.roi_rect_id)
            self.roi_rect_id = None

    def on_mouse_drag(self, event):
        if self.original_image is None or self.roi_start is None:
            return
        x0, y0 = self.roi_start
        x1, y1 = self._clamp_to_image_area(event.x, event.y)

        box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if self.roi_rect_id is None:
            self.roi_rect_id = self.canvas.create_rectangle(
                *box, outline=ACCENT, width=2
            )
        else:
            # Move the existing rectangle in place instead of deleting and
            # recreating it on every mouse-motion event — noticeably
            # smoother while dragging.
            self.canvas.coords(self.roi_rect_id, *box)
        self.roi_box_display = box

    def on_mouse_up(self, event):
        if self.original_image is None or self.roi_start is None:
            return
        x0, y0 = self.roi_start
        x1, y1 = self._clamp_to_image_area(event.x, event.y)
        self.roi_box_display = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.roi_start = None

    def on_clear_roi(self):
        self.roi_box_original = None
        self.roi_confirmed = False
        self.padded_crop_image = None
        self.padded_box_original = None
        self._redraw_confirmed_roi_overlay()  # clears any drawn rectangle
        self.preview_canvas.delete("all")
        self.lbl_roi_coords.config(text="—")
        self.lbl_padded_coords.config(text="—")
        self.lbl_padding.config(text="—")
        self.lbl_crop_size.config(text="—")
        self._reset_prediction_ui()

    def on_confirm_roi(self):
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please upload an ultrasound image first.")
            return
        if self.roi_box_display is None:
            messagebox.showwarning("No ROI", "Please draw a rectangle around the nodule first.")
            return

        x1c, y1c, x2c, y2c = self.roi_box_display
        if (x2c - x1c) <= 0 or (y2c - y1c) <= 0:
            messagebox.showwarning("Invalid ROI", "Selected ROI has zero width or height.")
            return

        # Translate canvas coordinates -> image-relative display coordinates
        # by removing the centering offset (the canvas fills its whole grid
        # cell, but the fitted image is only drawn centered within it).
        x1d = x1c - self.canvas_offset_x
        y1d = y1c - self.canvas_offset_y
        x2d = x2c - self.canvas_offset_x
        y2d = y2c - self.canvas_offset_y

        # Convert DISPLAY coordinates -> ORIGINAL image coordinates.
        # This is critical: the on-screen image may be scaled down from the
        # original resolution, so raw canvas pixels cannot be used directly.
        ox1, oy1 = display_to_original_coords(
            x1d, y1d, self.orig_w, self.orig_h, self.disp_w, self.disp_h
        )

        # Use ceil for the lower-right endpoint so the selected region is
        # represented faithfully in original-image coordinates.
        ox2 = int(np.ceil(
            min(max(float(x2d), 0.0), float(self.disp_w))
            * self.orig_w / self.disp_w
        ))
        oy2 = int(np.ceil(
            min(max(float(y2d), 0.0), float(self.disp_h))
            * self.orig_h / self.disp_h
        ))

        ox1 = max(0, min(ox1, self.orig_w - 1))
        oy1 = max(0, min(oy1, self.orig_h - 1))
        ox2 = max(1, min(ox2, self.orig_w))
        oy2 = max(1, min(oy2, self.orig_h))

        if ox2 <= ox1 or oy2 <= oy1:
            messagebox.showwarning(
                "Invalid ROI",
                "The selected ROI is too small after coordinate conversion."
            )
            return

        roi_original = (ox1, oy1, ox2, oy2)
        self.roi_box_original = roi_original

        try:
            padded_crop, padded_box = crop_with_padding(self.original_image, roi_original)
        except ValueError as exc:
            messagebox.showerror("ROI Error", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("ROI Error", f"Unexpected error while cropping ROI:\n{exc}")
            return

        self.padded_crop_image = padded_crop
        self.padded_box_original = padded_box
        self.roi_confirmed = True

        # Show the FINAL padded ROI on the source image so the user can
        # visually verify exactly what is being sent to the classifier.
        self._redraw_confirmed_roi_overlay()

        # Update preview: fit the COMPLETE crop inside the fixed preview
        # box, preserving its aspect ratio exactly (no cropping, no
        # stretching) regardless of whether it is wide, tall, tiny, or
        # large, then center it. This is a display-only transform — the
        # actual tensor sent to the model is produced independently below
        # via predict_nodule()/preprocess_input, unaffected by this sizing.
        preview_disp, pw, ph = fit_image_to_box(
            padded_crop, PREVIEW_DISPLAY_SIZE[0], PREVIEW_DISPLAY_SIZE[1],
            allow_upscale=True, max_upscale=4.0
        )
        self.tk_preview_image = ImageTk.PhotoImage(preview_disp)
        self.preview_canvas.delete("all")
        cw, ch = PREVIEW_DISPLAY_SIZE
        self.preview_canvas.create_image((cw - pw) // 2, (ch - ph) // 2, anchor="nw", image=self.tk_preview_image)

        # Update info labels
        self.lbl_roi_coords.config(
            text=f"({int(ox1)}, {int(oy1)}) → ({int(ox2)}, {int(oy2)})"
        )
        self.lbl_padded_coords.config(
            text=f"({padded_box[0]}, {padded_box[1]}) → ({padded_box[2]}, {padded_box[3]})"
        )
        self.lbl_padding.config(text=f"{int(PADDING_RATIO * 100)}%")
        crop_w = padded_box[2] - padded_box[0]
        crop_h = padded_box[3] - padded_box[1]
        self.lbl_crop_size.config(text=f"{crop_w} × {crop_h}")
        self.lbl_status.config(text="ROI confirmed — the padded nodule crop is ready for V3 analysis.")

        self._reset_prediction_ui()

    # ---- Prediction ----
    def on_analyze(self):
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please upload an ultrasound image first.")
            return
        if not self.roi_confirmed or self.padded_crop_image is None:
            messagebox.showwarning("No ROI Confirmed", "Please draw and confirm a nodule ROI first.")
            return
        if not self.model_handle.is_ready:
            messagebox.showerror("Model Unavailable", self.model_handle.load_error or "Model is not loaded.")
            return

        self.btn_analyze.config(state="disabled")
        self.result_banner.config(bg=BG_PANEL_ALT)
        self._result_inner.config(bg=BG_PANEL_ALT)
        self.result_icon.config(text="◔", bg=BG_PANEL_ALT, fg=ACCENT)
        self.result_text.config(text="ANALYZING…", bg=BG_PANEL_ALT, fg=ACCENT)
        self.lbl_status.config(text="Running V3 inference on the padded nodule crop…")

        # Run the EXISTING predict_nodule() pipeline unchanged, just on a
        # background thread so the UI does not freeze during TensorFlow
        # inference. The already-loaded model (loaded exactly once at
        # startup) is reused as-is; only the call site is asynchronous.
        crop_for_inference = self.padded_crop_image
        worker = threading.Thread(
            target=self._run_prediction_worker, args=(crop_for_inference,), daemon=True
        )
        worker.start()
        self._pred_poll_job = self.after(40, self._poll_prediction_queue)

    def _run_prediction_worker(self, crop_image: Image.Image):
        """Runs on a background thread. Must NOT touch any Tkinter widget
        directly — results are handed back via the thread-safe queue and
        applied to the UI on the main thread by _poll_prediction_queue."""
        try:
            result = predict_nodule(crop_image, self.model_handle)
            self._pred_queue.put(("ok", result))
        except Exception as exc:  # noqa: BLE001
            self._pred_queue.put(("error", exc))

    def _poll_prediction_queue(self):
        """Runs on the Tkinter main thread via after(). Picks up the
        worker thread's result (if ready yet) and updates the UI."""
        try:
            status, payload = self._pred_queue.get_nowait()
        except queue.Empty:
            self._pred_poll_job = self.after(40, self._poll_prediction_queue)
            return

        self._pred_poll_job = None

        if status == "ok":
            result = payload
            self._display_result(result)
            self.lbl_status.config(text="Analysis complete.")
            if self.log_enabled.get():
                self._log_current_prediction(result)
        else:
            exc = payload
            traceback.print_exc()
            messagebox.showerror("Prediction Error", f"Prediction failed:\n{exc}")
            self._reset_prediction_ui()

        self.btn_analyze.config(state="normal")

    def _display_result(self, result: dict):
        is_malignant = result["prediction"] == "Malignant"
        color = BAD if is_malignant else GOOD
        icon = "⚠" if is_malignant else "✔"
        self.result_banner.config(bg=color)
        self._result_inner.config(bg=color)
        self.result_icon.config(text=icon, bg=color, fg="#0a0e13")
        self.result_text.config(text=result["prediction"].upper(), bg=color, fg="#0a0e13")

        self.lbl_prob.config(text=f"{result['probability'] * 100:.1f}%")
        self.lbl_conf.config(text=f"{result['confidence'] * 100:.1f}%")

    def _log_current_prediction(self, result: dict):
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image_filename": Path(self.original_path).name if self.original_path else "",
            "orig_width": self.orig_w,
            "orig_height": self.orig_h,
            "roi_x1": int(self.roi_box_original[0]) if self.roi_box_original else "",
            "roi_y1": int(self.roi_box_original[1]) if self.roi_box_original else "",
            "roi_x2": int(self.roi_box_original[2]) if self.roi_box_original else "",
            "roi_y2": int(self.roi_box_original[3]) if self.roi_box_original else "",
            "padded_x1": self.padded_box_original[0] if self.padded_box_original else "",
            "padded_y1": self.padded_box_original[1] if self.padded_box_original else "",
            "padded_x2": self.padded_box_original[2] if self.padded_box_original else "",
            "padded_y2": self.padded_box_original[3] if self.padded_box_original else "",
            "crop_width": (self.padded_box_original[2] - self.padded_box_original[0]) if self.padded_box_original else "",
            "crop_height": (self.padded_box_original[3] - self.padded_box_original[1]) if self.padded_box_original else "",
            "malignant_probability": f"{result['probability']:.6f}",
            "prediction": result["prediction"],
            "threshold": result["threshold"],
            "model_version": "TN5000_EfficientNetB0_V3_NoduleCrop_final",
        }
        log_prediction(record)


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main():
    model_handle = ModelHandle()
    app = ThyroidApp(model_handle)
    app.mainloop()


if __name__ == "__main__":
    main()
