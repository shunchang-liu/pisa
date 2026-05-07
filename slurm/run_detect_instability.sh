#!/bin/bash
#SBATCH --job-name=detect_instability
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100:1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
cd $SLURM_SUBMIT_DIR

# Reward model used to generate perturbations
# Options: PKU-Alignment/beaver-7b-v2.0-reward | Skywork/Skywork-Reward-V2-Llama-3.1-8B
#          Skywork/Skywork-Reward-V2-Qwen3-4B  | ethz-spylab/poisoned-reward-7b-SUDO-10
MODEL_NAME="PKU-Alignment/beaver-7b-v2.0-reward"

# Path to trained SAE checkpoint
SAE_PATH="${SAE_CHECKPOINT}"

# RM layer whose activations are encoded by the SAE (paper: 12)
LAYER=12

# Feature type for the classifier input (paper: both)
# Options: both (SAE + raw hidden states) | sae | raw
FEATURE_TYPE="both"

# Fraction of data used for training the MLP classifier (paper: 0.7)
TRAIN_RATIO=0.7

# Output directory for saved classifier
OUTPUT_DIR="./classifier_models"

# --- Single dataset ---
DATASET_PATH="./perturbation_results/beaver-2-7b_tqa_gradient_raw.json"

python src/detect_instability.py \
    --model_name    "${MODEL_NAME}" \
    --layer         ${LAYER} \
    --sae_path      "${SAE_PATH}" \
    --feature_type  "${FEATURE_TYPE}" \
    --train_ratio   ${TRAIN_RATIO} \
    --dataset_path  "${DATASET_PATH}" \
    --output_dir    "${OUTPUT_DIR}"

# --- Multiple datasets (pooled before split) ---
# python detect_instability.py \
#     --model_name    "${MODEL_NAME}" \
#     --layer         ${LAYER} \
#     --sae_path      "${SAE_PATH}" \
#     --feature_type  "${FEATURE_TYPE}" \
#     --train_ratio   ${TRAIN_RATIO} \
#     --dataset_paths \
#         ./perturbation_results/beaver-2-7b_tqa_gradient_raw.json \
#         ./perturbation_results/beaver-2-7b_tqa_inject_raw.json \
#     --output_dir    "${OUTPUT_DIR}"
