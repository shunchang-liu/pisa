#!/bin/bash
#SBATCH --job-name=raw_feature_steering
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
cd $SLURM_SUBMIT_DIR

# Reward model to defend (no SAE required for this method)
# Options: PKU-Alignment/beaver-7b-v2.0-reward | Skywork/Skywork-Reward-V2-Llama-3.1-8B
#          Skywork/Skywork-Reward-V2-Qwen3-4B  | ethz-spylab/poisoned-reward-7b-SUDO-10
MODEL_NAME="ethz-spylab/poisoned-reward-7b-SUDO-10"

# RM layer to extract hidden states from (paper: 12)
LAYER=12

# Scale factor β applied to the steering vector at inference time (paper: 5)
# Paper values: 1 | 5 | 10 | 15
STEERING_STRENGTH=5

# Fraction of data used to compute the steering vector (remainder is test; paper: 0.7)
TRAIN_RATIO=0.7

# Run mode: train_and_test | train_only | test_only
MODE="train_and_test"

# Output directory
OUTPUT_DIR="./mitigation_results"

python src/raw_feature_steering.py \
    --model_name        "${MODEL_NAME}" \
    --layer             ${LAYER} \
    --steering_strength ${STEERING_STRENGTH} \
    --train_ratio       ${TRAIN_RATIO} \
    --mode              "${MODE}" \
    --output_dir        "${OUTPUT_DIR}" \
    --dataset_paths \
        ./perturbation_results/llama-7b-poisoned_tqa_gradient_raw.json \
        ./perturbation_results/llama-7b-poisoned_tqa_inject_raw.json \
        ./perturbation_results/llama-7b-poisoned_tqa_backdoor_raw.json
