#!/usr/bin/env python3
import os
import sys

def main():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        from huggingface_hub import login
        login(token=token)

    repo_id = os.environ.get("HF_REPO", "Zenith754/goodreads-bert-classifier")
    src_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(src_dir))
    os.execv(sys.executable, [sys.executable, os.path.join(src_dir, "eval.py"),
                               "--model_path", repo_id, "--mode", "hub"])

if __name__ == "__main__":
    main()
