#!/bin/bash
#SBATCH --job-name=analyze_features
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
cd $SLURM_SUBMIT_DIR

# Reward model to analyze; set MODEL_TAG to a short identifier used in output file names
# Options:
#   PKU-Alignment/beaver-7b-v2.0-reward    → MODEL_TAG="beaver-2-7b"
#   Skywork/Skywork-Reward-V2-Llama-3.1-8B → MODEL_TAG="llama-3-8b"
#   Skywork/Skywork-Reward-V2-Qwen3-4B     → MODEL_TAG="qwen-3-4b"
#   ethz-spylab/poisoned-reward-7b-SUDO-10 → MODEL_TAG="llama-7b-poisoned"
MODEL_NAME="PKU-Alignment/beaver-7b-v2.0-reward"
MODEL_TAG="beaver-2-7b"

# Path to trained SAE checkpoint
SAE_PATH="${SAE_CHECKPOINT}"

# RM layer whose SAE features are analyzed (paper: 12)
LAYER=12

# Perturbation dataset to analyze
DATASET_PATH="./perturbation_results/beaver-2-7b_tqa_gradient_raw.json"

# Short tag for the dataset, used in output file names
DATASET_TAG="beaver-2-7b_tqa_gradient"

# Root directory for analysis outputs
RESULTS_ROOT="./feature_analysis"

python src/analyze_features.py \
    --model_name    "${MODEL_NAME}" \
    --model_tag     "${MODEL_TAG}" \
    --sae_path      "${SAE_PATH}" \
    --layer         ${LAYER} \
    --dataset_path  "${DATASET_PATH}" \
    --dataset_tag   "${DATASET_TAG}" \
    --results_root  "${RESULTS_ROOT}"
