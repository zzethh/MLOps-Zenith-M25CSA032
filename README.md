# Lab 2 – CIFAR-10 CNN with FLOPs and Training Dynamics  
**Course:** CSL7120 – ML / DL / Ops  
**Name:** Zenith  
**Roll Number:** M25CSA032  
**Institute:** IIT Jodhpur  

---

## Overview

This repository contains the submission for **Lab 2 – Worksheet 1**, where a
Convolutional Neural Network (CNN) is trained on the **CIFAR-10** dataset using
a **custom dataloader**. The assignment focuses on model analysis beyond
accuracy, including **FLOPs computation**, **gradient flow**, and
**weight update flow**, with all metrics and visualizations logged to
**Weights & Biases (W&B)**.

---

## Dataset

- **CIFAR-10**
  - 50,000 training images
  - 10,000 test images
  - Image size: 32 × 32 × 3
  - 10 classes

A custom dataset class `CustomCIFAR10` wraps
`torchvision.datasets.CIFAR10` and applies dataset-specific transforms.

**Transforms**
- Training:
  - Random crop (32, padding=4)
  - Random horizontal flip
  - Normalization (CIFAR-10 statistics)
- Testing:
  - Normalization only

---

## Model Architecture

The selected model is a lightweight **SimpleCNN**, consisting of:

- 4 convolutional blocks  
  `(Conv2D → BatchNorm → ReLU → MaxPool)`
  - Channels: 32 → 64 → 128 → 128
- Fully connected layers:
  - `128 × 8 × 8 → 512 → 10`
- Dropout: 0.5 before the final classifier

This architecture is well-suited for CIFAR-10 resolution and balances accuracy
with computational cost.

---

## Training Configuration

- **Epochs:** 30  
- **Optimizer:** Adam  
- **Learning rate:** 0.001  
- **Loss function:** Cross-Entropy  
- **Batch size:** 128  
- **Device:** CUDA (if available), otherwise CPU  

Training and evaluation metrics are logged to W&B at every epoch.

---

## FLOPs Analysis

- FLOPs are computed using forward-pass hooks over all `Conv2D` and `Linear`
  layers.
- Input shape: `(1, 3, 32, 32)`
- FLOPs formula:
  - Convolution: `2 × Cin × Cout × Kh × Kw × Hout × Wout`
  - Linear: `2 × in_features × out_features`

**Result:**
- **0.1612 GFLOPs per sample**

This value represents the computational cost of a single forward pass and is
logged to W&B.

---

## Results

### Final Performance
- **Final test accuracy:** **84.54%**
- Training loss decreased steadily from ~1.79 to ~0.52.
- Validation accuracy improved consistently and stabilized after ~25 epochs.

### Training Dynamics
- No severe overfitting observed.
- Validation accuracy closely tracks training accuracy.

---

## Gradient and Weight Flow

To analyze training stability:

- **Gradient flow**
  - Mean and maximum gradient magnitudes per layer
  - Logged after backpropagation
- **Weight flow**
  - Mean and maximum weight magnitudes per layer
  - Logged during training

**Observations**
- No vanishing or exploding gradients.
- Stable weight magnitudes across layers.
- Healthy learning dynamics throughout training.

All gradient and weight flow visualizations are available in W&B.

---

## Visualizations (W&B)

The following are logged and available via the W&B workspace:

- Training & validation loss curves
- Training & validation accuracy curves
- Gradient flow (layer-wise)
- Weight flow (layer-wise)
- Confusion matrix (test set)
- Per-class accuracy
- Sample predictions (correct vs incorrect)

---

## Experiment Tracking (Weights & Biases)

All experiments and visualizations were logged using Weights & Biases.

**W&B Project Link:**  
https://wandb.ai/lab72343/lab2_cifar10

Logged artifacts include:
- Training and validation loss/accuracy curves
- Gradient flow plots
- Weight update flow plots
- Confusion matrix
- Sample predictions

---

## Repository Structure

```text
name_roll_number_lab2_worksheet1/
│
├── codes/        # Training, model, FLOPs, and visualization code
├── figures/      # Local copies of generated figures (if any)
├── logs/         # Training logs
├── report/       # Final PDF report
└── README.md     # This file
