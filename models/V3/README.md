# Final V3 Model

`TN5000_EfficientNetB0_V3_NoduleCrop_final.keras` is the trained EfficientNetB0 V3 nodule-crop classifier used by the desktop application.

Expected application input:

- RGB image crop
- 10% padded nodule ROI
- resized to 224×224
- EfficientNetB0 `preprocess_input`

Decision threshold: **0.55**.

The model is a research/educational artifact and is not a clinical diagnostic device.
