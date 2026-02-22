import argparse
import os
import json
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from data import prepare_datasets
from utils import compute_metrics, plot_confusion_matrix, plot_per_class_bars, plot_eval_metrics_bar
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
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Running evaluation...")
    metrics = trainer.evaluate()
    print("Evaluation Results:", metrics)

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_{args.mode}" if args.mode else ""
    results_file = os.path.join(output_dir, f"eval_results{suffix}.json")
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Results saved to {results_file}")

    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    true_labels = predictions_output.label_ids
    label_names = [id2label[i] for i in range(len(id2label))]

    plot_confusion_matrix(
        true_labels, preds, label_names,
        os.path.join(output_dir, f"confusion_matrix{suffix}.png"),
    )

    report_dict = classification_report(true_labels, preds, target_names=label_names, output_dict=True)
    plot_per_class_bars(report_dict, label_names, os.path.join(output_dir, f"per_class_metrics{suffix}.png"))

    plot_eval_metrics_bar(metrics, os.path.join(output_dir, f"overall_metrics{suffix}.png"))

    report_str = classification_report(true_labels, preds, target_names=label_names)
    print("\nClassification Report:\n")
    print(report_str)

    with open(os.path.join(output_dir, f"classification_report{suffix}.txt"), "w") as f:
        f.write(report_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="local", choices=["local", "hub"],
                        help="Evaluation mode: local or hub")
    args = parser.parse_args()
    main(args)
