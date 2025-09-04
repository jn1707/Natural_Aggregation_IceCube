#!/bin/bash
#SBATCH --job-name=transformer_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gr10_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=08:00:00

# Create the logs directory if it doesn't exist
mkdir -p logs

# Load CUDA module
module load cuda/12.8

# Print GPU specs to log
nvidia-smi

# Activate your virtual environment
source /groups/icecube/jniko/Natural_Aggregation_IceCube/.venv/bin/activate

# Set environment variables for PyTorch/FlashAttention debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run your training script
python /groups/icecube/jniko/Natural_Aggregation_IceCube/02_Adverserial_Network/training.py