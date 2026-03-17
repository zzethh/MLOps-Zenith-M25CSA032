# Assignment 4 — Transformer Tuning (English → Hindi)

**Student**: Zenith (`M25CSA032`)  
**Course**: CSL7590 — Deep Learning  

This folder contains the baseline run and the Ray Tune + Optuna hyperparameter tuning run for the provided from-scratch Transformer translation model (English → Hindi).

## What’s included (submission checklist)

- **Tuned implementation (Ray Tune + Optuna)**: `m25csa032_ass_4_tuned_en_to_hi.py`  
- **Logs + metrics**: `logs/`
  - Baseline: `logs/baseline_logs_30062.txt` (100 epochs), `logs/baseline_logs_30174.txt` (40 epochs)  
  - Tuning: `logs/tuning_logs_30166.txt`, `logs/best_config_and_metrics.json`
- **Report (PDF)**: `m25csa032_ass_4_report.pdf`
- **Best tuned weights**: uploaded to Google Drive as `m25csa032_ass_4_best_model.pth`  
  - Link: `https://drive.google.com/drive/folders/1eLTWTpjH66O-VclpRte-40JA_lSYWNUF?usp=sharing`

## Key results (from logs in this folder)

### Baseline (100 epochs, `logs/baseline_logs_30062.txt`)

- **Training time**: 56.89 minutes  
- **Final loss (@epoch 100)**: 0.0975  
- **BLEU (NLTK, printed as ×100)**: 71.47 (raw BLEU ≈ 0.7147)

### Final rerun with best tuned hyperparameters (40 epochs, `logs/baseline_logs_30174.txt`)

- **Epoch budget**: 40  
- **Hyperparameters**: `batch_size=32`, `num_heads=8`, `d_ff=2048`, `dropout=0.1495`, `lr=1.23e-4`  
- **Final loss (@epoch 40)**: 0.1764  
- **Training time**: 22.44 minutes  
- **BLEU (NLTK, printed as ×100)**: 94.57 (raw BLEU ≈ 0.9457)

### Ray Tune + Optuna (15 epochs per trial, `logs/tuning_logs_30166.txt`)

- **Best trial ID**: `train_tune_49cd135f`  
- **Best config (as printed in logs)**:
  - `lr`: 0.000122888
  - `batch_size`: 32
  - `dropout`: 0.14951
  - `d_ff`: 2048
  - `num_heads`: 8
- **Best trial metrics (15 epochs)**:
  - `loss`: 0.94976
  - `bleu` (raw): 0.57471 (≈ 57.47 when ×100)
  - `time_total_s`: 87.18 s

> Note: The baseline script prints BLEU as `score * 100`, while Ray Tune logs store the raw BLEU in `[0, 1]`. The tuned script now reports both `bleu` (raw) and `bleu_100` (×100) to keep things comparable.

## How to run

### Install dependencies

```bash
pip install --user -r requirements.txt
```

### Run tuning sweep (Ray Tune + Optuna)

```bash
python3 m25csa032_ass_4_tuned_en_to_hi.py
```

This will run 20 trials (OptunaSearch) capped at 15 epochs each, and then retrain the best config for 15 epochs and save:

- `logs/best_config_and_metrics.json`
- `m25csa032_ass_4_best_model.pth` (also uploaded to the Drive link above)

## File notes / consistency

- `logs/baseline_logs_30062.txt` corresponds to the **100-epoch baseline**.
- `logs/baseline_logs_30174.txt` is a **40-epoch rerun** using the best tuned hyperparameters.

