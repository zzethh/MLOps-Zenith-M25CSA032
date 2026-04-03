#!/bin/bash
#SBATCH --job-name=hw5-q1-vit-lora
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --time=34:00:00
#SBATCH --output=logs/q1_logs_%j.txt

module purge
module load python/3.10.pytorch

pip install --user -U accelerate "peft<0.11" optuna wandb "transformers<4.40" adversarial-robustness-toolbox

export WANDB_API_KEY="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"

echo "Starting Q1 Execution..."
cd /scratch/m25csa032/dlops_ass5/q1_vit_lora

python3 train.py --run_baseline
python3 train.py --run_optuna

echo "Q1 Job Complete!"
