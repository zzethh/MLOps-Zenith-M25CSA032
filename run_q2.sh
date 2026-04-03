#!/bin/bash
#SBATCH --job-name=hw5-q2-adv-art
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --time=34:00:00
#SBATCH --output=logs/q2_logs_%j.txt

module purge
module load python/3.10.pytorch

pip install --user -U adversarial-robustness-toolbox wandb

export WANDB_API_KEY="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"

echo "Starting Q2 Execution..."
cd /scratch/m25csa032/dlops_ass5/q2_adv_attacks

python3 train_classifier.py
python3 attack_fgsm.py
python3 train_detector.py

echo "Q2 Job Complete!"
