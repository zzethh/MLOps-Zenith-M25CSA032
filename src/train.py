import os
os.environ["WANDB_DISABLED"] = "true"
import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from data import prepare_datasets 
from utils import compute_metrics 

def main(args):
    train_dataset, test_dataset, label2id, id2label, tokenizer = prepare_datasets(model_id=args.model_id)
    print(f"Loading Model: {args.model_id}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=2, label2id=label2id, id2label=id2label
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none", 
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./final_model")
    parser.add_argument("--model_id", type=str, default="distilbert-base-cased")
    args = parser.parse_args()
    main(args)
