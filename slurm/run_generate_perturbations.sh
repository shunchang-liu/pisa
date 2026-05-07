#!/bin/bash
#SBATCH --job-name=generate_perturbations
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
cd $SLURM_SUBMIT_DIR

# Reward model to perturb
# Options: PKU-Alignment/beaver-7b-v2.0-reward | Skywork/Skywork-Reward-V2-Llama-3.1-8B
#          Skywork/Skywork-Reward-V2-Qwen3-4B  | ethz-spylab/poisoned-reward-7b-SUDO-10
MODEL_NAME="Skywork/Skywork-Reward-V2-Qwen3-4B"

# Input preference pairs (jsonl with 'chosen' and 'rejected' fields)
INPUT_FILE="./harmless-test.jsonl"

# Perturbation method: gradient | inject | backdoor
#   gradient — gradient-guided paraphrasing via GPT-4o (requires OPENAI_API_KEY; paper: max_steps=15, topk=5, temperature=0.7)
#   inject   — predefined pattern injection, no API key needed
#   backdoor — fixed trigger phrase appended to rejected response, no API key needed
PERTURBATION_METHOD="gradient"

# Output directory for generated perturbation files
OUTPUT_DIR="./perturbation_results"

# Experiment tag used to name the output file (<tag>_raw.json)
EXPERIMENT_NAME="qwen3-4b_tqa_gradient"

python src/generate_perturbations.py \
    --model_name        "${MODEL_NAME}" \
    --input_file        "${INPUT_FILE}" \
    --attack_method     "${PERTURBATION_METHOD}" \
    --output_dir        "${OUTPUT_DIR}" \
    --experiment_name   "${EXPERIMENT_NAME}" \
    --api_key           "${OPENAI_API_KEY}"
