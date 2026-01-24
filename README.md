# DL-Ops Assignment 1: Deep Learning & Hardware Benchmarking

**Name:** Zenith  
**Roll Number:** M25CSA032  
**Department:** Computer Science  

---

## 📌 Abstract
This repository contains the benchmarks for ResNet architectures against Classical SVMs and analyzes hardware acceleration (CPU vs. GPU) on the FashionMNIST dataset. Experiments were conducted on the **DPU-GPU HPC Cluster** (Intel Xeon Gold 6326 + NVIDIA A30). 

We utilized a **70-10-20** train-val-test split and **Mixed Precision Training (AMP)**.

---

## ⚙️ System Specifications
- **CPU:** Dual Intel Xeon Gold 6326 (32 Cores @ 2.90 GHz)
- **GPU:** 2x NVIDIA A30 Tensor Core GPUs (24GB VRAM)
- **RAM:** 256 GB DDR4
- **Networking:** NVIDIA BlueField-2 DPU

---

## 📊 Q1 (A): ResNet Hyperparameter Tuning Results
Analysis of ResNet-18 and ResNet-50 on MNIST and FashionMNIST with varying Batch Sizes (BS), Optimizers, and Learning Rates (LR).

| Dataset | Model | BS | Opt | LR | Epochs | PinMem | Test Acc (%) | Time (s) |
|---|---|---|---|---|---|---|---|---|
| MNIST | ResNet-18 | 16 | SGD | 0.001 | 4 | True | 99.24% | 174.54s |
| MNIST | ResNet-50 | 16 | SGD | 0.001 | 4 | True | 99.12% | 220.15s |
| MNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | True | **99.20%** | 149.36s |
| MNIST | ResNet-18 | 32 | Adam | 0.001 | 4 | True | 99.04% | 105.06s |
| FashionMNIST | ResNet-18 | 16 | SGD | 0.001 | 4 | True | 89.45% | 179.80s |
| FashionMNIST | ResNet-50 | 16 | SGD | 0.001 | 4 | True | 88.10% | 235.60s |
| FashionMNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | True | **91.12%** | 160.20s |
| FashionMNIST | ResNet-50 | 16 | Adam | 0.001 | 4 | True | 89.90% | 240.50s |

> **Observation:** ResNet-18 with Adam Optimizer generally achieved the best balance of accuracy and convergence speed.

---

## 📈 Q1 (B): SVM Classification Results
Benchmarking classical Machine Learning (SVM) kernels against Deep Learning.

| Dataset | Kernel | Test Acc (%) | Time (ms) |
|---|---|---|---|
| MNIST | Poly | 97.71% | 169,357 ms |
| MNIST | RBF | **97.92%** | 162,137 ms |
| FashionMNIST | Poly | 86.30% | 279,310 ms |
| FashionMNIST | RBF | **88.28%** | 222,573 ms |

> **Observation:** While SVM (RBF) achieves respectable accuracy on MNIST, the training time is orders of magnitude higher than ResNet, proving it inefficient for complex image datasets.

---

## 🚀 Q2: Hardware Acceleration (CPU vs GPU)
Benchmarking training time and GFLOPs on FashionMNIST (10 Epochs) to measure hardware efficiency.

| Device | Model | Optimizer | Total Time (s) | Final Acc (%) | GFLOPs |
|---|---|---|---|---|---|
| **CPU** | ResNet-18 | Adam | 3161.05s | 92.74% | 0.2961 |
| **GPU (A30)** | ResNet-18 | Adam | **780.54s** | 92.83% | 0.2961 |
| **CPU** | ResNet-34 | Adam | 5502.84s | 92.51% | 0.5981 |
| **GPU (A30)** | ResNet-34 | Adam | 1140.24s | 92.81% | 0.5981 |
| **CPU** | ResNet-50 | Adam | 7221.72s | 92.72% | 0.6673 |
| **GPU (A30)** | ResNet-50 | Adam | 1432.99s | 92.19% | 0.6673 |

### Speedup Analysis
- **ResNet-18:** ~4.0x Speedup
- **ResNet-34:** ~4.8x Speedup
- **ResNet-50:** ~5.0x Speedup

> **Conclusion:** The NVIDIA A30 GPU provides a massive performance boost (4x-5x) over the Dual Xeon CPUs. The speedup factor increases with model complexity (ResNet-50 > ResNet-18), indicating better GPU utilization for compute-bound tasks.

---

## 📂 Repository Structure