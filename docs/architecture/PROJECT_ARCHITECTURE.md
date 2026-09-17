# Project Architecture

## Data analytics path

TN5000 → validation → duplicate/leakage analysis → leakage-safe split → feature extraction → EDA → statistical analysis

## AI path

Leakage-safe TN5000 → nodule crop generation → EfficientNetB0 transfer learning → validation threshold selection → fixed held-out test evaluation

## Application path

Full ultrasound → manual ROI → 10% padding → resize 224×224 → EfficientNetB0 preprocessing → V3 classifier → probability → threshold 0.55 → result
