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
- **CPU:** Dual Intel Xeon Gold 6326 (32 Cores @ 2.90 GHz)
- **GPU:** 2× NVIDIA A30 Tensor Core GPUs (24 GB VRAM each)
- **RAM:** 256 GB DDR4
- **Networking:** NVIDIA BlueField-2 DPU

---

## 🧪 Experimental Settings (Global)
- **Data Split:** 70% Train / 10% Validation / 20% Test  
- **Framework:** PyTorch  
- **AMP:** Enabled  

### Training Parameters
- **Epochs:**  
  - Q1(a): **4 and 10**  
  - Q2: **10**  

- **pin_memory:**  
  - Enabled (**True**) for main experiments  
  - Disabled (**False**) for ablation analysis  

---

## 📊 Q1 (A): ResNet Hyperparameter Tuning Results

> **Representative best configurations shown for clarity**  
> (Full logs available in Colab)

### Consolidated Results (Epochs & pin_memory Variations)

| Dataset | Model | BS | Optimizer | LR | Epochs | PinMem | Test Acc (%) | Time (s) |
|---|---|---|---|---|---|---|---|---|
| MNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | True | 99.24 | 174.54 |
| MNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | False | 99.27 | 298.57 |
| MNIST | ResNet-18 | 16 | Adam | 0.001 | 10 | True | **99.41** | 401.63 |
| MNIST | ResNet-18 | 16 | Adam | 0.001 | 10 | False | 99.41 | 729.87 |
| FashionMNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | True | 91.57 | 147.15 |
| FashionMNIST | ResNet-18 | 16 | Adam | 0.001 | 4 | False | 92.06 | 295.59 |
| FashionMNIST | ResNet-18 | 16 | Adam | 0.001 | 10 | True | **92.80** | 357.84 |
| FashionMNIST | ResNet-18 | 16 | Adam | 0.001 | 10 | False | 92.49 | 739.60 |
| MNIST | ResNet-50 | 32 | Adam | 0.001 | 10 | True | **99.28** | 611.36 |
| MNIST | ResNet-50 | 32 | Adam | 0.001 | 10 | False | 99.40 | 1242.54 |
| FashionMNIST | ResNet-50 | 16 | Adam | 0.001 | 10 | True | **92.91** | 990.71 |
| FashionMNIST | ResNet-50 | 16 | Adam | 0.001 | 10 | False | 91.95 | 1240.52 |

### Key Observations
- Increasing epochs (**4 → 10**) improves accuracy by ~0.3–0.5% at ~2× training time.
- **pin_memory=True** provides ~1.8–2× faster training with negligible accuracy impact.
- **ResNet-18** consistently offers the best accuracy–time trade-off.

---

## 📈 Q1 (B): SVM Classification Results (CPU)

| Dataset | Kernel | Test Acc (%) | Training Time (ms) |
|---|---|---|---|
| MNIST | Polynomial | 97.71 | 169,357 |
| MNIST | RBF | **97.92** | 162,137 |
| FashionMNIST | Polynomial | 86.30 | 279,310 |
| FashionMNIST | RBF | **88.28** | 222,573 |

**Observation:**  
SVMs achieve reasonable accuracy but are **orders of magnitude slower** than deep learning
models and do not scale well.

---

## 🚀 Q2: Hardware Acceleration (CPU vs GPU)

> **Dataset:** FashionMNIST  
> **Epochs:** 10  
> **pin_memory:** True  
> **AMP:** Enabled  

### CPU vs GPU Performance (SGD & Adam)

| Device | Model | Optimizer | Total Time (s) | Final Acc (%) | GFLOPs |
|---|---|---|---|---|---|
| CPU | ResNet-18 | SGD | 2724.24 | 91.61 | 0.2961 |
| CPU | ResNet-18 | Adam | 3161.05 | 92.74 | 0.2961 |
| GPU (A30) | ResNet-18 | SGD | **707.64** | 92.42 | 0.2961 |
| GPU (A30) | ResNet-18 | Adam | 780.54 | **92.83** | 0.2961 |
| CPU | ResNet-34 | SGD | 4684.85 | 91.80 | 0.5981 |
| CPU | ResNet-34 | Adam | 5502.84 | 92.51 | 0.5981 |
| GPU (A30) | ResNet-34 | SGD | **1088.74** | 91.17 | 0.5981 |
| GPU (A30) | ResNet-34 | Adam | 1140.24 | **92.81** | 0.5981 |
| CPU | ResNet-50 | SGD | 6284.05 | 91.41 | 0.6673 |
| CPU | ResNet-50 | Adam | 7221.72 | 92.72 | 0.6673 |
| GPU (A30) | ResNet-50 | SGD | **1265.14** | 91.17 | 0.6673 |
| GPU (A30) | ResNet-50 | Adam | 1432.99 | **92.19** | 0.6673 |

### Speedup Summary
- **ResNet-18:** ~4.0×  
- **ResNet-34:** ~4.8×  
- **ResNet-50:** ~5.0×  

**Conclusion:**  
GPU acceleration provides massive speedups for both SGD and Adam optimizers.  
Deeper models benefit more, confirming that GPUs are most effective for **compute-bound**
deep learning workloads.

---

