# 3D Vision Mamba with Grad-CAM++ Explainability for Multi-Stage Alzheimer's Disease Classification from Structural MRI

A 3D deep learning pipeline for classifying Alzheimer's Disease stages from structural MRI scans, built on the OASIS neuroimaging dataset. The project covers the full pipeline: data preprocessing, patient-level train/test splitting, class-imbalance-aware training, and model explainability through activation-based heatmaps.

## Overview

This project classifies brain MRI volumes into four stages of Alzheimer's Disease:
- **Non Demented**
- **Very Mild Dementia**
- **Mild Dementia**
- **Moderate Dementia**

The model is a 3D CNN built with Mamba-inspired residual blocks operating on volumetric MRI data, trained end-to-end with PyTorch.

## Results

**Final Test Accuracy: 80.28%**

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Non Demented     | 0.85      | 0.94   | 0.89     | 54      |
| Very Mild        | 0.55      | 0.50   | 0.52     | 12      |
| Mild             | 0.00      | 0.00   | 0.00     | 4       |
| Moderate         | 0.00      | 0.00   | 0.00     | 1       |
| **Accuracy**     |           |        | **0.80** | 71      |
| Macro Avg        | 0.35      | 0.36   | 0.35     | 71      |
| Weighted Avg     | 0.74      | 0.80   | 0.77     | 71      |

## Dataset

- **Source:** OASIS-based Alzheimer's MRI dataset (Kaggle)
- **Total patients:** 347, split at the patient level (no data leakage across train/test)
- **Class distribution (patients):**
  - Non Demented: 265
  - Very Mild Dementia: 58
  - Mild Dementia: 21
  - Moderate Dementia: 2

> **Note on class imbalance:** Moderate Dementia is represented by only 2 patients in the entire dataset, and Mild Dementia by 21. This is a known, dataset-level limitation — not a methodology issue — and significantly limits achievable performance on these two classes. Their metrics in the results table above should be interpreted with this in mind.

## Project Structure

\`\`\`
3D-Vision-Mamba-for-Alzheimer-s-MRI-Classification/
│
├── checkpoints/
│   └── vision_mamba_SUBMIT_FINAL.pth      # Final trained model weights
│
├── models/
│   ├── vision_mamba.py                    # 3D CNN / Mamba-inspired architecture
│   └── alzheimer_model.py
│
├── src/
│   ├── dataset_class.py                   # Dataset loader, preprocessing, augmentation
│   ├── train_mamba.py                     # Training script
│   ├── check_checkpoint.py                # Evaluate a saved checkpoint
│   ├── confusion_matrix.py                # Generate confusion matrix + classification report
│   ├── generate_heatmap.py                # Activation-based heatmap visualization
│   └── ... (data exploration / utility scripts)
│
├── split.json                              # Patient-level train/test split
└── README.md
\`\`\`

## Methodology

1. **Patient-level train/test split** — ensures no patient's scans appear in both train and test sets, preventing data leakage.
2. **Class-weighted loss** — `CrossEntropyLoss` with weights inversely proportional to class frequency (capped for the smallest class to avoid training instability).
3. **3D data augmentation** — random flips, small rotations, and intensity jitter applied during training only.
4. **Depth normalization** — MRI volumes are zero-padded or trimmed to a fixed depth (61 slices) for consistent batch shapes.
5. **Gradient clipping + cosine LR scheduling** — for training stability.
6. **Model selection by macro-F1** (over the statistically meaningful classes) rather than raw accuracy, to avoid bias toward the majority class.

## Key Findings / Debugging Notes

- An initial bug caused both Moderate Dementia patients to land in the training set, meaning the class was never evaluated. This was identified and fixed by rebalancing the split.
- A `WeightedRandomSampler` approach was tested to address class imbalance but caused training instability (the model would collapse toward over-predicting minority classes). Class-weighted loss alone proved more stable and was used in the final model.

## How to Run

\`\`\`bash
# Train the model
cd src
python train_mamba.py

# Evaluate the saved checkpoint
python check_checkpoint.py

# Generate confusion matrix and classification report
python confusion_matrix.py

# Generate explainability heatmaps
python generate_heatmap.py
\`\`\`

## Requirements

\`\`\`
torch
opencv-python
numpy
scikit-learn
matplotlib
\`\`\`

## Limitations & Future Work

- **Moderate Dementia (n=2 patients)** and **Mild Dementia (n=21 patients)** remain difficult to classify reliably due to severe data scarcity — more data collection or transfer learning from a larger pretrained 3D model would likely help most.
- The current architecture uses Mamba-inspired residual convolutional blocks rather than a true Selective State Space Model (SSM) with selective scan; implementing genuine bidirectional SSM blocks is a natural next step.
- Training was conducted on CPU; GPU access would allow higher input resolution and longer training, both likely to improve performance further.