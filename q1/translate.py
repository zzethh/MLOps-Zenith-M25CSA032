import torch
from transformers import MarianMTModel, MarianTokenizer
import sacrebleu

model_name = "Helsinki-NLP/opus-mt-bn-en"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

f = open("input.txt", "r", encoding="utf-8")
raw = f.read()
f.close()

lines = raw.strip().split("\n")

sentences = []
current = ""
for line in lines:
    text = line.strip()
    if text.startswith("#") or text == "":
        if current:
            sentences.append(current)
            current = ""
        continue
    if current:
        current = current + " " + text
    else:
        current = text
    if text.endswith(("।", ".", "?", "!")):
        sentences.append(current)
        current = ""
if current:
    sentences.append(current)

print("Total sentences:", len(sentences))

outputs = []
for i, text in enumerate(sentences):
    tokens = tokenizer([text], return_tensors="pt", padding=True).to(device)
    translated = model.generate(**tokens)
    tgt = tokenizer.decode(translated[0], skip_special_tokens=True)
    outputs.append(tgt)
    print(f"{i+1}: {tgt}")

f = open("output.txt", "w", encoding="utf-8")
for line in outputs:
    f.write(line + "\n")
f.close()

print("\nFirst statement:", outputs[0])

ref_file = open("reference.txt", "r", encoding="utf-8")
ref_raw = ref_file.read()
ref_file.close()

ref_sentences = []
current = ""
for line in ref_raw.strip().split("\n"):
    text = line.strip()
    if text.startswith("#") or text == "":
        if current:
            ref_sentences.append(current)
            current = ""
        continue
    if current:
        current = current + " " + text
    else:
        current = text
    if text.endswith((".", "?", "!")):
        ref_sentences.append(current)
        current = ""
if current:
    ref_sentences.append(current)

print("\nReference sentences:", len(ref_sentences))

min_len = min(len(outputs), len(ref_sentences))
bleu = sacrebleu.corpus_bleu(outputs[:min_len], [ref_sentences[:min_len]])
print("BLEU Score:", bleu.score)

f = open("bleu_score.txt", "w")
f.write(f"BLEU Score: {bleu.score}\n")
f.close()
