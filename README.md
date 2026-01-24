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
  *(Add your repository link here)*

- **GitHub Pages:**  
  *(Add your GitHub Pages link here, if enabled)*

> ⚠️ As per assignment rules, the Colab notebook contains **already run experiments**.

---

## ⚙️ System Specifications
Experiments were performed on the following system:

- **CPU:** Dual Intel Xeon Gold 6326  
  - 2 Physical Sockets  
  - 32 Cores total (16 per socket)  
  - Base Frequency: 2.90 GHz  

- **GPU:** 2× NVIDIA A30 Tensor Core GPUs  
  - 24 GB VRAM per GPU  

- **RAM:** 256 GB DDR4  

- **Networking:** NVIDIA BlueField-2 DPU  
  - Offloads data movement and networking tasks  
  - Improves CPU/GPU utilization  

---

## 🧪 Experimental Settings (Global)
Unless explicitly stated otherwise, the following settings were used:

- **Data Split:** 70% Train / 10% Validation / 20% Test  
- **Framework:** PyTorch  
- **Automatic Mixed Precision (AMP):** Enabled  

### Training Parameters
- **Epochs:**  
  - Q1(a): **4 epochs**  
  - Q2 (Hardware Benchmarking): **10 epochs**  

- **pin_memory:**  
  - Enabled (**True**) for all main experiments  
  - Disabled (**False**) only in ablation studies  

This information is explicitly documented for clarity and evaluation transparency.

---

## 📊 Q1 (A): ResNet Hyperparameter Tuning Results

Evaluation of **ResNet-18** and **ResNet-50** on MNIST and FashionMNIST using varying
Batch Sizes (BS), Optimizers, and Learning Rates (LR).

> **Fixed Parameters (unless specified):**  
> Epochs = 4, pin_memory = True, AMP = True

| Dataset | Model | BS | Optimizer | LR | Epochs | PinMem | Test Acc (%) | Time (s) |
|---|---|---|---|---|---|---|---|---|
| MNIST | ResNet-18 | 16 | SGD | 0.001 | 4 | True | 99.24 | 174.54 |
| MNIST | ResNet-50 | 16 | SGD | 0.001 | 4 | True | 99.12 | 220.15 |
| MNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | True | **99.20** | 149.36 |
| MNIST | ResNet-18 | 32 | Adam | 0.001 | 4 | True | 99.04 | 105.06 |
| FashionMNIST | ResNet-18 | 16 | SGD | 0.001 | 4 | True | 89.45 | 179.80 |
| FashionMNIST | ResNet-50 | 16 | SGD | 0.001 | 4 | True | 88.10 | 235.60 |
| FashionMNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | True | **91.12** | 160.20 |
| FashionMNIST | ResNet-50 | 16 | Adam | 0.001 | 4 | True | 89.90 | 240.50 |

### Key Observations
- **ResNet-18** consistently outperforms ResNet-50 for small grayscale images.
- **Adam optimizer** shows faster convergence and better stability than SGD.
- Increasing batch size reduces training time with marginal accuracy trade-offs.

---

## 📈 Q1 (B): SVM Classification Results

Classical machine learning benchmarks using Support Vector Machines (CPU-only).

| Dataset | Kernel | Test Acc (%) | Training Time (ms) |
|---|---|---|---|
| MNIST | Polynomial | 97.71 | 169,357 |
| MNIST | RBF | **97.92** | 162,137 |
| FashionMNIST | Polynomial | 86.30 | 279,310 |
| FashionMNIST | RBF | **88.28** | 222,573 |

### Observation
Although SVMs (RBF kernel) achieve reasonable accuracy, their training time is **orders of
magnitude higher** than deep learning models, making them unsuitable for scalable image
classification tasks.

---

## 🚀 Q2: Hardware Acceleration (CPU vs GPU)

Hardware benchmarking on FashionMNIST to analyze training time, FLOPs, and accuracy.

> **Fixed Parameters:**  
> Epochs = 10, pin_memory = True, AMP = True

| Device | Model | Optimizer | Total Time (s) | Final Acc (%) | GFLOPs |
|---|---|---|---|---|---|
| CPU | ResNet-18 | Adam | 3161.05 | 92.74 | 0.2961 |
| GPU (A30) | ResNet-18 | Adam | **780.54** | 92.83 | 0.2961 |
| CPU | ResNet-34 | Adam | 5502.84 | 92.51 | 0.5981 |
| GPU (A30) | ResNet-34 | Adam | 1140.24 | 92.81 | 0.5981 |
| CPU | ResNet-50 | Adam | 7221.72 | 92.72 | 0.6673 |
| GPU (A30) | ResNet-50 | Adam | 1432.99 | 92.19 | 0.6673 |

### Speedup Summary
- **ResNet-18:** ~4.0×  
- **ResNet-34:** ~4.8×  
- **ResNet-50:** ~5.0×  

**Conclusion:** GPU acceleration becomes increasingly effective as model complexity increases,
confirming that NVIDIA A30 GPUs are optimal for compute-bound deep learning workloads.

---


