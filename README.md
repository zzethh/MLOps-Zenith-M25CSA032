# DL-Ops Assignment 1: Deep Learning & Hardware Benchmarking

**Name:** Zenith  
**Roll Number:** M25CSA032  
**Department:** Computer Science  

---

## 📌 Abstract
This repository presents a comprehensive benchmarking study of deep learning and classical
machine learning models on the MNIST and FashionMNIST datasets. We evaluate **ResNet-18**
and **ResNet-50** architectures against classical **Support Vector Machines (SVMs)** and
analyze the impact of **hardware acceleration (CPU vs GPU)**.

All experiments were conducted on the **DPU–GPU HPC Cluster** (Dual Intel Xeon Gold 6326 +
NVIDIA A30 GPUs). A **70–10–20 train–validation–test split** and **Automatic Mixed Precision
(AMP)** were used for all deep learning experiments.

---

## 🔗 Submission Links
- **Google Colab Notebook (Executed):**  
  https://colab.research.google.com/drive/1WvjjTIT6QyN_ygidWWWAq5B-FjpUng4_?usp=sharing

- **GitHub Repository:**  
  https://github.com/zzethh/MLOps-Zenith-M25CSA032.git

- **GitHub Pages:**  
  https://zzethh.github.io/MLOps-Zenith-M25CSA032/

---

## ⚙️ System Specifications
- **CPU:** Dual Intel Xeon Gold 6326  
  - 2 Physical sockets  
  - 32 cores total (16 per socket)  
  - Base frequency: 2.90 GHz  

- **GPU:** 2× NVIDIA A30 Tensor Core GPUs  
  - 24 GB VRAM per GPU  

- **RAM:** 256 GB DDR4  

- **Networking:** NVIDIA BlueField-2 DPU  
  - Data-path offloading for improved throughput  

---

## ⚙️ Experimental Settings
- **Dataset Split:** 70% Train / 10% Validation / 20% Test  
- **Framework:** PyTorch  
- **Automatic Mixed Precision (AMP):** Enabled  

### Training Parameters
- **Epochs:** 4 and 10  
- **pin_memory:** True and False  

---

## 📊 Q1 (A): ResNet Hyperparameter Benchmarking

All combinations of the following parameters were evaluated:

- Dataset ∈ {MNIST, FashionMNIST}  
- Model ∈ {ResNet-18, ResNet-50}  
- Batch Size ∈ {16, 32}  
- Optimizer ∈ {SGD, Adam}  
- Learning Rate ∈ {0.001, 0.0001}  
- Epochs ∈ {4, 10}  
- pin_memory ∈ {True, False}  

### Complete Results (All Runs)

| Dataset | Model | BS | Optimizer | LR | Epochs | PinMem | Test Acc (%) | Time (s) |
|--------|-------|----|-----------|----|--------|--------|--------------|----------|
| MNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | True | 99.24 | 174.54 |
| MNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | False | 99.27 | 298.57 |
| MNIST | ResNet-18 | 16 | Adam | 0.001 | 10 | True | **99.41** | 401.63 |
| MNIST | ResNet-18 | 16 | Adam | 0.001 | 10 | False | 99.41 | 729.87 |
| MNIST | ResNet-18 | 32 | SGD | 0.001 | 4 | True | 98.95 | 186.42 |
| MNIST | ResNet-18 | 32 | SGD | 0.001 | 10 | True | 99.12 | 382.71 |
| MNIST | ResNet-50 | 32 | Adam | 0.001 | 10 | True | **99.28** | 611.36 |
| FashionMNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | True | 91.57 | 147.15 |
| FashionMNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | False | 92.06 | 295.59 |
| FashionMNIST | ResNet-18 | 16 | Adam | 0.001 | 10 | True | **92.80** | 357.84 |
| FashionMNIST | ResNet-18 | 16 | Adam | 0.001 | 10 | False | 92.49 | 739.60 |
| FashionMNIST | ResNet-50 | 16 | Adam | 0.001 | 10 | True | **92.91** | 990.71 |
| FashionMNIST | ResNet-50 | 16 | Adam | 0.001 | 10 | False | 91.95 | 1240.52 |

**Key Observations**
- Increasing epochs from **4 → 10** improves accuracy by ~0.3–0.5%.
- **pin_memory=True** reduces data-loading overhead, giving ~2× speedup.
- **ResNet-18** provides the best accuracy–time trade-off for small images.

---

## 📈 Training Dynamics & Convergence Analysis

### MNIST – ResNet-18 (10 Epochs, pin_memory=True)
![MNIST Convergence](figures/1a_MNIST_Training_Dynamics.png)

### FashionMNIST – ResNet-18 (10 Epochs, pin_memory=True)
![FashionMNIST Convergence](figures/1a_FashionMNIST_Training_Dynamics.png)

**Observation**
- Training loss decreases smoothly, indicating stable optimization.
- Validation accuracy saturates after ~6–7 epochs.
- No overfitting is observed.

---

## 📊 Q1 (B): SVM Classification Results (CPU Only)

| Dataset | Kernel | Test Acc (%) | Training Time (ms) |
|--------|--------|--------------|--------------------|
| MNIST | Polynomial | 97.71 | 169,357 |
| MNIST | RBF | **97.92** | 162,137 |
| FashionMNIST | Polynomial | 86.30 | 279,310 |
| FashionMNIST | RBF | **88.28** | 222,573 |

**Conclusion:**  
SVMs achieve reasonable accuracy but are significantly slower and do not scale well for
large image datasets.

---

## 🚀 Q2: Hardware Acceleration (CPU vs GPU)

### Adam Optimizer (10 Epochs, pin_memory=True)

| Device | Model | Time (s) | Final Acc (%) |
|-------|-------|----------|----------------|
| CPU | ResNet-18 | 3161.05 | 92.74 |
| GPU (A30) | ResNet-18 | **780.54** | 92.83 |
| CPU | ResNet-34 | 5502.84 | 92.51 |
| GPU (A30) | ResNet-34 | **1140.24** | 92.81 |
| CPU | ResNet-50 | 7221.72 | 92.72 |
| GPU (A30) | ResNet-50 | **1432.99** | 92.19 |

### SGD Optimizer (10 Epochs, pin_memory=True)

| Device | Model | Time (s) | Final Acc (%) |
|-------|-------|----------|----------------|
| CPU | ResNet-18 | 2724.24 | 91.61 |
| GPU (A30) | ResNet-18 | **707.64** | 92.42 |
| CPU | ResNet-34 | 4684.85 | 91.80 |
| GPU (A30) | ResNet-34 | **1088.74** | 91.17 |
| CPU | ResNet-50 | 6284.05 | 91.41 |
| GPU (A30) | ResNet-50 | **1265.14** | 91.17 |

**Speedup Summary**
- ResNet-18: ~4×  
- ResNet-34: ~4.8×  
- ResNet-50: ~5×  

---

