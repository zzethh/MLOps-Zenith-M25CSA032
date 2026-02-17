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

    accuracy = load_accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    precision = load_precision.compute(predictions=predictions, references=labels)[
        "precision"
    ]
    recall = load_recall.compute(predictions=predictions, references=labels)["recall"]

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(10, 8))
    sns.set(style="ticks", font_scale=1.2)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Purples", linewidths=1)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_training_history(output_dir, output_path):
    state_file = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_file):
        print("No trainer_state.json found. Skipping loss plot.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(state_file, "r") as f:
        data = json.load(f)

    history = data.get("log_history", [])
    if not history:
        return

    train_loss = [x["loss"] for x in history if "loss" in x]
    steps = [x["step"] for x in history if "loss" in x]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, label="Training Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Training history plot saved to {output_path}")


def save_misclassifications(y_true, y_pred, dataset, tokenizer, id2label, output_path):
    print(f"Saving misclassifications to {output_path}...")
    with open(output_path, "w") as f:
        f.write("TRUE LABEL\tPREDICTED LABEL\tREVIEW TEXT\n")
        f.write("=" * 100 + "\n")

        count = 0
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if true != pred:
                true_lbl = id2label[true]
                pred_lbl = id2label[pred]

                input_ids = dataset[i]["input_ids"]
                text = tokenizer.decode(input_ids, skip_special_tokens=True)

                clean_text = text[:150].replace("\n", " ")
                f.write(f"{true_lbl}\t{pred_lbl}\t{clean_text}...\n")
                count += 1
    print(f"Saved {count} misclassified examples.")
