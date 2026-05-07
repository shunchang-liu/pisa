#!/bin/bash
#SBATCH --job-name=sae_residual_correction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
cd $SLURM_SUBMIT_DIR

# Reward model to defend
# Options: PKU-Alignment/beaver-7b-v2.0-reward | Skywork/Skywork-Reward-V2-Llama-3.1-8B
#          Skywork/Skywork-Reward-V2-Qwen3-4B  | ethz-spylab/poisoned-reward-7b-SUDO-10
MODEL_NAME="ethz-spylab/poisoned-reward-7b-SUDO-10"

# Feature representation to use for the correction head
# Options: sae (SAE latent space) | raw (raw hidden states, no SAE needed)
FEATURE_TYPE="sae"

# Path to trained SAE checkpoint (required when FEATURE_TYPE=sae)
SAE_PATH="${SAE_CHECKPOINT}"

# RM layer to extract features from (paper: 12)
LAYER=12

# Training epochs for the correction head (paper: 100 for OOD eval)
# Paper values: 100 | 200 | 300 | 400
NUM_EPOCHS=100

# Learning rate for the correction head optimizer (paper: 1e-3)
LEARNING_RATE=1e-3

# L2 regularization weight that penalizes non-zero corrections on benign samples (paper: 0.05)
REG_WEIGHT=0.05

# Fraction of data used for training (remainder is test; paper: 0.7)
TRAIN_RATIO=0.7

# Output directory for saved correction head
OUTPUT_DIR="./mitigation_results"

python src/sae_residual_correction.py \
    --model_name    "${MODEL_NAME}" \
    --feature_type  "${FEATURE_TYPE}" \
    --sae_path      "${SAE_PATH}" \
    --layer         ${LAYER} \
    --num_epochs    ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --reg_weight    ${REG_WEIGHT} \
    --train_ratio   ${TRAIN_RATIO} \
    --output_dir    "${OUTPUT_DIR}" \
    --dataset_paths \
        ./perturbation_results/llama-7b-poisoned_tqa_gradient_raw.json \
        ./perturbation_results/llama-7b-poisoned_tqa_inject_raw.json \
        ./perturbation_results/llama-7b-poisoned_tqa_backdoor_raw.json
