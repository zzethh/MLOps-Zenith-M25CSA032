# Lab 2 – Worksheet 1  
## CIFAR-10 CNN Training and Analysis  
**Name:** Zenith  
**Roll Number:** M25CSA032  

---

## Objective
The objective of this lab is to design and train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset, analyze its computational complexity, and study training dynamics using gradient flow and weight update visualizations. All metrics and visualizations are logged using Weights & Biases (W&B).

---

## Dataset
- **Dataset:** CIFAR-10  
- **Classes:** 10  
- **Image Size:** 32 × 32 × 3  
- A **custom PyTorch Dataset and DataLoader** were implemented instead of using built-in loaders.

---

## Model Architecture
- A custom CNN with multiple convolutional blocks followed by fully connected layers.
- ReLU activations and max-pooling were used.
- The model balances accuracy and efficiency.

**Model Complexity:**  
- FLOPs: **0.1612 GFLOPs**

---

## Training Details
- **Epochs:** 30  
- **Optimizer:** Adam  
- **Loss Function:** Cross-Entropy Loss  
- **Batch Size:** As specified in code  
- **Hardware:** GPU (when available)

---

## Results

### Performance
- **Final Training Accuracy:** ~82.76%  
- **Final Validation Accuracy:** ~84.54%  
- **Final Test Accuracy:** **84.54%**

Training and validation losses steadily decrease, indicating stable convergence without severe overfitting.

---

## Visualization and Analysis
All the following were logged and visualized using **Weights & Biases**:

- Training loss vs epochs  
- Validation loss vs epochs  
- Training accuracy vs epochs  
- Validation accuracy vs epochs  
- Gradient flow across layers  
- Weight update magnitudes across epochs  

These visualizations help in understanding:
- Gradient stability during training  
- Effective learning across layers  
- Smooth convergence behavior  

---

## Observations
- Gradient flow remained stable throughout training, with no evidence of vanishing or exploding gradients.
- Weight updates were larger during early epochs and gradually stabilized, indicating proper convergence.
- The model achieved a good accuracy–compute trade-off given its low FLOPs.

---


## Repository Structure

```text
zenith_M25CSA032_lab2_worksheet/
├── codes/        # Model, dataset, training, and utility scripts
├── figures/      # Generated plots and visualizations
├── logs/         # Training logs
├── report/       # PDF report
└── README.md     # This file
