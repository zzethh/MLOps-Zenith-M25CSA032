import random
import requests
import gzip
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer


def load_reviews(url, head=10000, sample_size=1000):
    reviews = []
    count = 0
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Warning: Failed to fetch {url}")
        return []
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))


def prepare_datasets(model_id="distilbert-base-cased"):
    print(f"Loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    genre_url_dict = {
        'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
        'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
        'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
        'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
        'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
        'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
        'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
        'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
    }

    print("Downloading and processing Goodreads dataset...")
    train_texts, train_labels_str = [], []
    test_texts, test_labels_str = [], []

    random.seed(42)

    for genre, url in genre_url_dict.items():
        print(f"Loading reviews for {genre}...")
        _reviews = load_reviews(url, head=10000, sample_size=1000)
        split_idx = int(0.8 * len(_reviews))
        for _review in _reviews[:split_idx]:
            train_texts.append(_review)
            train_labels_str.append(genre)
        for _review in _reviews[split_idx:]:
            test_texts.append(_review)
            test_labels_str.append(genre)

    unique_labels = sorted(list(set(train_labels_str)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    train_labels = [label2id[l] for l in train_labels_str]
    test_labels = [label2id[l] for l in test_labels_str]

    train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'labels': test_labels})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing dataset...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_test = tokenized_test.remove_columns(["text"])

    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    tokenized_train = tokenized_train.shuffle(seed=42)
    tokenized_test = tokenized_test.shuffle(seed=42)

    return tokenized_train, tokenized_test, label2id, id2label, tokenizer
