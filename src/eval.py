import argparse
import os
import json
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from data import prepare_datasets
from utils import compute_metrics, plot_confusion_matrix
from sklearn.metrics import classification_report

def main(args):
    # 1. Prepare Data
    _, test_dataset, label2id, id2label, _ = prepare_datasets(model_id="distilbert-base-cased")

    # 2. Load Model
    print(f"Loading trained model from {args.model_path}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except OSError:
        print(f"Error: Could not load model from {args.model_path}.")
        return

    # 3. Setup Trainer
    trainer = Trainer(
        model=model,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 4. Run Evaluation
    print("Running evaluation...")
    metrics = trainer.evaluate()
    print("Evaluation Results:", metrics)
    
    # --- FIX: ALWAYS SAVE TO 'results' FOLDER ---
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save Metrics
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 5. Generate Predictions for Confusion Matrix
    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    true_labels = predictions_output.label_ids
    label_names = [id2label[i] for i in range(len(label2id))]

    # Save Confusion Matrix
    plot_confusion_matrix(true_labels, preds, label_names, os.path.join(output_dir, "confusion_matrix.png"))
    
    # Print Report
    report = classification_report(true_labels, preds, target_names=label_names)
    print("\nClassification Report:\n")
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
