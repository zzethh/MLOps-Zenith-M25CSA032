import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import evaluate
import json
import os
from sklearn.metrics import confusion_matrix


def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    load_precision = evaluate.load("precision")
    load_recall = evaluate.load("recall")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    precision = load_precision.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    recall = load_recall.compute(predictions=predictions, references=labels, average="weighted")["recall"]

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(12, 9))
    sns.set(style="ticks", font_scale=1.1)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Purples", linewidths=1)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix — Goodreads Genre Classification")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_per_class_bars(report_dict, label_names, output_path):
    precision = [report_dict[l]["precision"] for l in label_names]
    recall = [report_dict[l]["recall"] for l in label_names]
    f1 = [report_dict[l]["f1-score"] for l in label_names]

    x = np.arange(len(label_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label="Precision", color="#5B8DB8")
    ax.bar(x, recall, width, label="Recall", color="#E07B4F")
    ax.bar(x + width, f1, width, label="F1-Score", color="#6AAB7A")

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall and F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in label_names], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Per-class metrics bar chart saved to {output_path}")


def plot_eval_metrics_bar(metrics, output_path):
    keys = ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]
    labels = ["Accuracy", "F1 (weighted)", "Precision (weighted)", "Recall (weighted)"]
    values = [metrics.get(k, 0) for k in keys]
    colors = ["#5B8DB8", "#6AAB7A", "#E07B4F", "#C47AC0"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Overall Evaluation Metrics")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Overall metrics bar chart saved to {output_path}")


def plot_comparison_bars(local_metrics, hub_metrics, output_path):
    keys = ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]
    labels = ["Accuracy", "F1", "Precision", "Recall"]
    local_vals = [local_metrics.get(k, 0) for k in keys]
    hub_vals = [hub_metrics.get(k, 0) for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, local_vals, width, label="Local Model", color="#5B8DB8")
    b2 = ax.bar(x + width/2, hub_vals, width, label="Hub Model", color="#E07B4F")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Local vs. Hub Model Evaluation Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.75)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison bar chart saved to {output_path}")
