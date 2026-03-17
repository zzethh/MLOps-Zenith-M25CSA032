import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import ray
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

import json
import time

df = pd.read_csv('English-Hindi.tsv', sep='\t', header=None, names=["id1", "en", "id2", "hi"])
df = df[["en", "hi"]].dropna().reset_index(drop=True)
df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx = 4

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in sentence.lower().strip().split():
                frequencies[word] += 1
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

    def numericalize(self, sentence):
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in sentence.lower().strip().split()]

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])

en_vocab = Vocabulary(freq_threshold=2)
hi_vocab = Vocabulary(freq_threshold=2)
en_vocab.build_vocab(df["en"].tolist())
hi_vocab.build_vocab(df["hi"].tolist())

def encode_sentence(sentence, vocab, max_len=50):
    tokens = [vocab.stoi["<sos>"]] + vocab.numericalize(sentence)[:max_len-2] + [vocab.stoi["<eos>"]]
    return tokens + [vocab.stoi["<pad>"]] * (max_len - len(tokens))

class TranslationDataset(Dataset):
    def __init__(self, df, en_vocab, hi_vocab, max_len=50):
        self.en_sentences = df["en"].tolist()
        self.hi_sentences = df["hi"].tolist()
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        src = encode_sentence(self.en_sentences[idx], self.en_vocab, self.max_len)
        tgt = encode_sentence(self.hi_sentences[idx], self.hi_vocab, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    return src_batch, tgt_batch[:, :-1], tgt_batch[:, 1:]

global_dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=50)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.query_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(self.dropout(attention_weights), V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attention_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=50, dropout=0.1):
        super().__init__()
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_subsequent_mask(self, size, device):
        return torch.tril(torch.ones((size, size))).bool().to(device)

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        device = src.device
        src_mask = self.make_pad_mask(src, src_pad_idx)
        tgt_pad_mask = self.make_pad_mask(tgt, tgt_pad_idx)
        tgt_sub_mask = self.make_subsequent_mask(tgt.size(1), device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        enc_out = self.dropout(self.pos_enc(self.encoder_embed(src)))
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        dec_out = self.dropout(self.pos_enc(self.decoder_embed(tgt)))
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        return self.fc_out(dec_out)

def evaluate_bleu_nltk(model, dataset, en_vocab, hi_vocab, device, max_len=50):
    model.eval()
    references, hypotheses = [], []
    smoothie = SmoothingFunction().method4

    for en_sentence, hi_sentence in dataset:
        tokens = encode_sentence(en_sentence, en_vocab, max_len)
        src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        tgt_tokens = [hi_vocab["<sos>"]]
        
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(src_tensor, tgt_tensor, en_vocab["<pad>"], hi_vocab["<pad>"])
            next_token = output[0, -1].argmax().item()
            tgt_tokens.append(next_token)
            if next_token == hi_vocab["<eos>"]:
                break
                
        pred = ' '.join([hi_vocab.itos[idx] for idx in tgt_tokens[1:-1]])
        references.append([hi_sentence.split()])
        hypotheses.append(pred.split())

    return corpus_bleu(references, hypotheses, smoothing_function=smoothie)

def train_single_config(
    config,
    dataset,
    *,
    max_epochs,
    device,
    max_len=50,
):
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=512,
        num_layers=6,
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_len=max_len,
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=hi_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    val_dataset = [
        ("I love you.", "मैं तुमसे प्यार करता हूँ।"),
        ("How are you?", "आप कैसे हैं?"),
        ("You should sleep.", "आपको सोना चाहिए।"),
    ]

    best = {"epoch": 0, "loss": float("inf"), "bleu": 0.0}
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for src, tgt_input, tgt_output in train_loader:
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

            output = model(src, tgt_input, en_vocab["<pad>"], hi_vocab["<pad>"])
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(train_loader))
        bleu = evaluate_bleu_nltk(model, val_dataset, en_vocab, hi_vocab, device, max_len=max_len)

        if avg_loss < best["loss"]:
            best = {"epoch": epoch, "loss": float(avg_loss), "bleu": float(bleu)}

    elapsed_minutes = (time.time() - start_time) / 60.0
    return model, best, elapsed_minutes

# ==========================================
# 4. RAY TUNE TRAINING FUNCTION
# ==========================================
def train_tune(config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=512, 
        num_layers=6,
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_len=50,
        dropout=config["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=hi_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    val_dataset = [
        ("I love you.", "मैं तुमसे प्यार करता हूँ।"),
        ("How are you?", "आप कैसे हैं?"),
        ("You should sleep.", "आपको सोना चाहिए।")
    ]

    MAX_TUNE_EPOCHS = 15
    
    for epoch in range(MAX_TUNE_EPOCHS):
        model.train()
        epoch_loss = 0

        for src, tgt_input, tgt_output in train_loader:
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

            output = model(src, tgt_input, en_vocab["<pad>"], hi_vocab["<pad>"])
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        
        current_bleu = evaluate_bleu_nltk(model, val_dataset, en_vocab, hi_vocab, device)

        tune.report({"loss": avg_loss, "bleu": current_bleu, "bleu_100": current_bleu * 100.0})

if __name__ == "__main__":
    print("Initializing Ray...")
    ray.init(num_gpus=1, ignore_reinit_error=True)

    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "dropout": tune.uniform(0.1, 0.4),
        "d_ff": tune.choice([1024, 2048]),
        "num_heads": tune.choice([4, 8])
    }

    optuna_search = OptunaSearch(metric="loss", mode="min")
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        max_t=15,
        grace_period=3,
        reduction_factor=2,
    )

    print("Starting Ray Tune + Optuna Sweep...")
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_tune, dataset=global_dataset),
            resources={"gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=20, # Number of different combinations it will try
        ),
        param_space=search_space,
    )

    results = tuner.fit()

    # Get and print the best result
    best_result = results.get_best_result(metric="loss", mode="min")
    print("\n" + "="*40)
    print("🎯 BEST HYPERPARAMETER CONFIGURATION FOUND:")
    print("="*40)
    for key, value in best_result.config.items():
        print(f" - {key}: {value}")
    print("="*40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model, best_metrics, elapsed_minutes = train_single_config(
        best_result.config,
        global_dataset,
        max_epochs=15,
        device=device,
        max_len=50,
    )

    weights_path = "m25csa032_ass_4_best_model.pth"
    torch.save(best_model.state_dict(), weights_path)

    summary = {
        "best_config": dict(best_result.config),
        "retrain_max_epochs": 15,
        "retrain_best_epoch": best_metrics["epoch"],
        "retrain_best_loss": best_metrics["loss"],
        "retrain_bleu_raw": best_metrics["bleu"],
        "retrain_bleu_x100": best_metrics["bleu"] * 100.0,
        "retrain_time_minutes": elapsed_minutes,
        "ray_best_result": {
            "metric": "loss",
            "mode": "min",
            "loss": float(best_result.metrics.get("loss")) if best_result.metrics else None,
            "bleu": float(best_result.metrics.get("bleu")) if best_result.metrics else None,
            "bleu_100": float(best_result.metrics.get("bleu_100")) if best_result.metrics else None,
        },
    }

    with open("best_config_and_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved best model weights to: {weights_path}")
    print("✅ Saved best config/metrics to: best_config_and_metrics.json")