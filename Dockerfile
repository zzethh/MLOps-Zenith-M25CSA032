FROM python:3.9-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV MODEL_NAME="Zenith754/goodreads-bert-classifier"

CMD ["sh", "-c", "python src/eval.py --model_path $MODEL_NAME"]