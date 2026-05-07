#!/bin/bash
#SBATCH --job-name=evaluate_rb2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
cd $SLURM_SUBMIT_DIR

# Reward model to evaluate
# Options: PKU-Alignment/beaver-7b-v2.0-reward | Skywork/Skywork-Reward-V2-Llama-3.1-8B
#          Skywork/Skywork-Reward-V2-Qwen3-4B  | ethz-spylab/poisoned-reward-7b-SUDO-10
MODEL_NAME="ethz-spylab/poisoned-reward-7b-SUDO-10"

# Path to trained SAE checkpoint (required for sae_fs and sae_rc methods)
SAE_PATH="${SAE_CHECKPOINT}"

# RM layer used by all methods (paper: 12)
LAYER=12

# Methods to evaluate (space-separated subset of: base raw_fs sae_fs sae_rc)
#   base    — original RM with no defense
#   raw_fs  — Raw Feature Steering baseline
#   sae_fs  — SAE Feature Steering
#   sae_rc  — SAE Residual Correction
METHODS="base raw_fs sae_fs sae_rc"

# Fraction of perturbation data used to train defenses (remainder is held out; paper: 0.7)
TRAIN_RATIO=0.7

# Raw Feature Steering: steering vector scale β (paper: 5)
# Paper values: 1 | 5 | 10 | 15
RAW_FS_BETA=5.0

# SAE Feature Steering: number of anomalous features to suppress (paper: 200)
SAE_FS_TOP_K=200
# SAE Feature Steering: suppression multiplier η (paper: -0.001)
# Paper values: -0.001 | -0.01 | -0.1 | -1.0
SAE_FS_ETA=-0.001

# SAE Residual Correction: training epochs (paper: 100 for OOD eval)
# Paper values: 100 | 200 | 300 | 400
SAE_RC_EPOCHS=100
# SAE Residual Correction: learning rate (paper: 1e-3)
SAE_RC_LR=1e-3
# SAE Residual Correction: L2 regularization on benign corrections (paper: 0.05)
SAE_RC_REG=0.05

# Output directory and optional tag for multi-dataset runs
OUTPUT_DIR="./rb2_results"
DATASET_TAG="poisoned7b_multi3_tqa"

python src/evaluate_rb2.py \
    --model_name    "${MODEL_NAME}" \
    --sae_path      "${SAE_PATH}" \
    --layer         ${LAYER} \
    --methods       ${METHODS} \
    --train_ratio   ${TRAIN_RATIO} \
    --raw_fs_beta   ${RAW_FS_BETA} \
    --sae_fs_top_k  ${SAE_FS_TOP_K} \
    --sae_fs_eta    ${SAE_FS_ETA} \
    --sae_rc_epochs ${SAE_RC_EPOCHS} \
    --sae_rc_lr     ${SAE_RC_LR} \
    --sae_rc_reg    ${SAE_RC_REG} \
    --output_dir    "${OUTPUT_DIR}" \
    --dataset_tag   "${DATASET_TAG}" \
    --save_artifacts \
    --dataset_paths \
        ./perturbation_results/llama-7b-poisoned_tqa_gradient_raw.json \
        ./perturbation_results/llama-7b-poisoned_tqa_inject_raw.json \
        ./perturbation_results/llama-7b-poisoned_tqa_backdoor_raw.json
