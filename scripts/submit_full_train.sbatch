#!/bin/bash
#SBATCH --job-name=dinov3_bigearthnet_full
#SBATCH --output=/home/carmenoliver/my_projects/dinov3-testing-stuff/outputs_full/slurm-%j.out
#SBATCH --error=/home/carmenoliver/my_projects/dinov3-testing-stuff/outputs_full/slurm-%j.err
#SBATCH --gres=gpu:1
# Set number of tasks so SLURM sets SLURM_NTASKS (required by the code's distributed init)
# For a single-GPU job this should be 1.
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

#SBATCH --partition=gpu

set -euo pipefail

export PYTHONPATH=/home/carmenoliver/my_projects/dinov3-testing-stuff
mkdir -p /home/carmenoliver/my_projects/dinov3-testing-stuff/outputs_full

# Activate conda env if available so environment variables (CUDA libs, PATH, etc.) are set
if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  . "$HOME/.conda/etc/profile.d/conda.sh"
  conda activate envr
fi

echo "Starting dinov3 full training on $(hostname) at $(date)"

/home/carmenoliver/.conda/envs/envr/bin/python \
  /home/carmenoliver/my_projects/dinov3-testing-stuff/dinov3/train/train.py \
  --config-file /home/carmenoliver/my_projects/dinov3-testing-stuff/dinov3/configs/train/train_bigearthnet_vits16.yaml \
  --output-dir /home/carmenoliver/my_projects/dinov3-testing-stuff/outputs_full 

  