#!/bin/bash
#SBATCH --job-name=sae_feature_steering
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

# Path to trained SAE checkpoint
SAE_PATH="${SAE_CHECKPOINT}"

# RM layer whose activations pass through the SAE (paper: 12)
LAYER=12

# Number of top anomalous SAE features to suppress (paper: 200)
TOP_K=200

# Suppression multiplier η applied to selected feature activations (paper: -0.001)
# Paper values: -0.001 | -0.01 | -0.1 | -1.0  (negative suppresses activation)
SUPPRESSION_FACTOR=-0.001

python src/sae_feature_steering.py \
    --model_name            "${MODEL_NAME}" \
    --sae_path              "${SAE_PATH}" \
    --layer                 ${LAYER} \
    --top_k                 ${TOP_K} \
    --suppression_factor    ${SUPPRESSION_FACTOR} \
    --dataset_paths \
        ./perturbation_results/llama-7b-poisoned_tqa_gradient_raw.json \
        ./perturbation_results/llama-7b-poisoned_tqa_inject_raw.json \
        ./perturbation_results/llama-7b-poisoned_tqa_backdoor_raw.json
