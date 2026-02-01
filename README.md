# Lab 2 – CIFAR-10 CNN Analysis with FLOPs and Gradient Flow

**Course:** CSL7120 – ML / DL / Ops  
**Student Name:** Zenith  
**Roll Number:** M25CSA032  
**Lab:** Lab 2 – Worksheet 1  

---

## Objective

The objective of this lab is to:
- Train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset
- Implement a custom PyTorch dataloader
- Compute FLOPs for the selected model
- Analyze training dynamics using gradient flow and weight update flow
- Log all experiments and visualizations using Weights & Biases (W&B)

---

## Model Architecture

A custom CNN architecture (`SimpleCNN`) was implemented, consisting of:
- Four convolutional layers with Batch Normalization and ReLU activation
- Max pooling layers for spatial downsampling
- Fully connected layers for final classification
- Cross-entropy loss for optimization

The architecture is lightweight and well-suited for CIFAR-10 (32×32 RGB images).

---

## Dataset

- **Dataset:** CIFAR-10  
- **Number of Classes:** 10  
- **Image Size:** 32×32 RGB  
- **Train/Test Split:** Standard CIFAR-10 split  

### Preprocessing and Augmentation
- Normalization using CIFAR-10 mean and standard deviation
- Random horizontal flip
- Random crop for data augmentation

A custom dataset class (`CustomCIFAR10`) was created by wrapping `torchvision.datasets.CIFAR10`.

---

## Training Configuration

- **Epochs:** 30  
- **Optimizer:** Adam  
- **Learning Rate:** 1e-3  
- **Loss Function:** CrossEntropyLoss  
- **Batch Size:** As defined in the training script  
- **Device:** GPU (if available)  

---

## FLOPs Analysis

FLOPs were calculated using forward hooks registered on convolutional and linear layers.

- **Total FLOPs per forward pass:** approximately **0.161 GFLOPs**

This provides insight into the computational complexity of the model relative to its performance.

---

## Gradient Flow and Weight Update Analysis

To analyze training stability and optimization behavior:
- Gradient flow was visualized using mean and maximum gradient magnitudes per layer
- Weight update flow was visualized using layer-wise weight magnitudes

### Observations
- Gradients remained stable across layers throughout training
- No evidence of vanishing or exploding gradients
- Weight updates were smooth, indicating stable optimization

---

## Results

- **Final Test Accuracy:** approximately 85–86%
- Certain classes exhibited higher confusion due to visual similarity
- The model demonstrated stable convergence with limited overfitting

Additional results include confusion matrices, per-class accuracy, and prediction visualizations.

---

## Experiment Tracking (Weights & Biases)

All experiments and visualizations were logged using Weights & Biases.

**W&B Project Link:**  
https://wandb.ai/m25csa032-iit-jodhpur/lab2_cifar10/workspace?nw=nwuserm25csa032

Logged artifacts include:
- Training and validation loss/accuracy curves
- Gradient flow plots
- Weight update flow plots
- Confusion matrix
- Sample predictions

---

## Repository Structure

```text
zenith_M25CSA032_lab2_worksheet/
├── codes/        # Model, dataset, training, and utility scripts
├── figures/      # Generated plots and visualizations
├── logs/         # Training logs
├── report/       # PDF report
└── README.md     # This file
