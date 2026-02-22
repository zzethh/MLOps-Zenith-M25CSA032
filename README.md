# MLOps Assignment 3: End-to-End HuggingFace Model Training & Docker Deployment

**Name:** Zenith | **Assessment Code:** M25CSA032 | **Course:** MLOps

## Project Overview

Fine-tuning `distilbert-base-cased` on the UCSD Goodreads dataset for 8-genre book review classification. The pipeline includes training, evaluation, HuggingFace Hub deployment, and Docker containerization.

## Links

- **HuggingFace Model:** [Zenith754/goodreads-bert-classifier](https://huggingface.co/Zenith754/goodreads-bert-classifier)
- **GitHub Repository:** [MLOps-Zenith-M25CSA032](https://github.com/zzethh/MLOps-Zenith-M25CSA032)

## Model Details

| Property | Value |
|---|---|
| Architecture | DistilBERT Base Cased |
| Task | Sequence Classification (Genre Classification) |
| Dataset | Goodreads Reviews (8 genres) |
| Accuracy | 55.4% |
| Weighted F1 | 0.547 |

**Genres:** Children, Comics & Graphic, Fantasy & Paranormal, History & Biography, Mystery/Thriller/Crime, Poetry, Romance, Young Adult

## Project Structure

```
├── Dockerfile              # Dev image (local model bundled)
├── Dockerfile.eval         # Production image (pulls from HF Hub)
├── requirements.txt
├── README.md
├── src/
│   ├── data.py             # Goodreads dataset streaming & tokenization
│   ├── train.py            # Training with HF Trainer API
│   ├── eval.py             # Evaluation (supports --mode local|hub)
│   ├── utils.py            # Metrics & visualization functions
│   └── hub_eval_entry.py   # Docker entrypoint for Hub evaluation
├── results/                # Evaluation outputs & visualizations
├── logs/                   # Execution logs
├── final_model/            # Trained model weights
└── report/                 # LaTeX report & compiled PDF
```

## Quick Start

### Prerequisites
- Docker
- Python 3.9+
- Git

### Option 1: Run Evaluation with Docker (Local Model)

```bash
# Clone the repository
git clone https://github.com/zzethh/MLOps-Zenith-M25CSA032.git
cd MLOps-Zenith-M25CSA032

# Build the Docker image
docker build -t mlops-assignment .

# Run evaluation (results saved to ./results/)
docker run --rm -v $(pwd)/results:/app/results mlops-assignment
```

### Option 2: Run Evaluation from HuggingFace Hub (Production)

```bash
# Build the production Docker image
docker build -f Dockerfile.eval -t mlops-eval .

# Run evaluation (downloads model from HF Hub)
docker run --rm \
    -e HF_TOKEN=<your_hf_token> \
    -e HF_REPO=Zenith754/goodreads-bert-classifier \
    -v $(pwd)/results:/app/results \
    mlops-eval
```

### Option 3: Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run local evaluation
python src/eval.py --model_path ./final_model --mode local

# Run Hub evaluation
python src/eval.py --model_path Zenith754/goodreads-bert-classifier --mode hub
```

## Evaluation Results

| Metric | Local | Hub | Diff |
|---|---|---|---|
| Accuracy | 0.5538 | 0.5538 | 0.0000 |
| Weighted F1 | 0.5471 | 0.5471 | 0.0000 |
| Weighted Precision | 0.5455 | 0.5455 | 0.0000 |
| Weighted Recall | 0.5538 | 0.5538 | 0.0000 |

### Output Files in `results/`

| File | Description |
|---|---|
| `eval_results_local.json` | Metrics from local model evaluation |
| `eval_results_hub.json` | Metrics from HuggingFace Hub model evaluation |
| `classification_report_local.txt` | Per-class report (local) |
| `classification_report_hub.txt` | Per-class report (hub) |
| `confusion_matrix_local.png` | 8×8 genre confusion matrix (local) |
| `confusion_matrix_hub.png` | 8×8 genre confusion matrix (hub) |
| `per_class_metrics_local.png` | Per-genre Precision/Recall/F1 bars (local) |
| `per_class_metrics_hub.png` | Per-genre Precision/Recall/F1 bars (hub) |
| `overall_metrics_local.png` | Overall Accuracy/F1/Precision/Recall bar (local) |
| `overall_metrics_hub.png` | Overall metrics bar (hub) |
| `comparison_local_hub.png` | Local vs Hub side-by-side comparison |
| `training_loss.png` | Eval loss & accuracy per epoch |

---

_Assignment 3 for MLOps Course — Zenith (M25CSA032)_
