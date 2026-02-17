# MLOps Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

**Assessment Code**: M25CSA032  
**Name**: Zenith  
**Course**: MLOps (DL-Ops)

## Project Overview

This project implements a complete machine learning workflow for a GoodReads BERT Classifier. It involves:

1.  Fine-tuning a `distilbert-base-cased` model on the IMDb dataset (as a proxy for sentiment analysis).
2.  Containerizing the training and evaluation workflow using Docker.
3.  Deploying the model artifacts to Hugging Face.
4.  Automating evaluation with a Python script.

## Model Details

- **Model Architecture**: DistilBERT Base Cased
- **Task**: Sequence Classification (Sentiment Analysis)
- **Dataset**: IMDb (Binary Sentiment: Positive/Negative)
- **Hugging Face Model Link**: [Zenith754/goodreads-bert-classifier](https://huggingface.co/Zenith754/goodreads-bert-classifier/tree/main)

## Setup & Installation

### Prerequisites

- Docker
- Python 3.9+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/zzethh/MLOps-Zenith-M25CSA032.git
cd MLOps-Zenith-M25CSA032
```

### 2. running with Docker

Build the Docker image:

```bash
docker build -t mlops-assignment .
```

Run the container (this will automatically run the evaluation script):

```bash
docker run mlops-assignment
```

You can also run it interactively to explore:

```bash
docker run -it mlops-assignment /bin/bash
```

### 3. Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run evaluation:

```bash
python src/eval.py --model_path "Zenith754/goodreads-bert-classifier"
```

## Evaluation Results

The model was evaluated on the test set with the following metrics:

| Metric        | Value  |
| :------------ | :----- |
| **Accuracy**  | 87.20% |
| **F1 Score**  | 87.60% |
| **Precision** | 83.70% |
| **Recall**    | 91.87% |
| **Runtime**   | 50.16s |

### Visuals

The evaluation process generates the following visuals in the `results/` directory:

- **Confusion Matrix**: Shows the distribution of true vs. predicted labels.
- **Training Loss**: Plots the training loss over steps.

---

_Assignment 3 for MLOps Course - Zenith (M25CSA032)_
