import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_datasets(model_id="distilbert-base-cased"):
    print(f"Loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Downloading dataset...")
    dataset = load_dataset("imdb")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
    test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    return train_dataset, test_dataset, label2id, id2label, tokenizer
