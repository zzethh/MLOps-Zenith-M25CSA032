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
    _, test_dataset, label2id, id2label, _ = prepare_datasets(
        model_id="distilbert-base-cased"
    )

    print(f"Loading trained model from {args.model_path}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except OSError:
        print(f"Error: Could not load model from {args.model_path}.")
        return

    trainer = Trainer(
        model=model,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Running evaluation...")
    metrics = trainer.evaluate()
    print("Evaluation Results:", metrics)

    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save trainer state for loss plotting
    trainer.save_state()

    # Generate Training Loss Plot
    from utils import plot_training_history

    plot_training_history(output_dir, os.path.join(output_dir, "training_loss.png"))

    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    true_labels = predictions_output.label_ids
    label_names = [id2label[i] for i in range(len(label2id))]

    plot_confusion_matrix(
        true_labels,
        preds,
        label_names,
        os.path.join(output_dir, "confusion_matrix.png"),
    )

    report = classification_report(true_labels, preds, target_names=label_names)
    print("\nClassification Report:\n")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
