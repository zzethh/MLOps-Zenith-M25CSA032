# Use a lightweight Python base image
FROM python:3.9-slim-bullseye

# Set working directory
WORKDIR /app

# Install git (needed for some HF dependencies)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Default command: Run evaluation using the model from Hugging Face
# We use an environment variable for the model name so it's flexible
ENV MODEL_NAME="Zenith754/goodreads-bert-classifier"

CMD ["sh", "-c", "python src/eval.py --model_path $MODEL_NAME"]