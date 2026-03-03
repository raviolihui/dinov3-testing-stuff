#!/bin/bash
#SBATCH --job-name=dinov3_vitl16_12b_v3
#SBATCH --partition=gpu
#SBATCH --output=/home/carmenoliver/my_projects/dinov3-testing-stuff/outputs_vitl12_v3/slurm-%j.out
#SBATCH --error=/home/carmenoliver/my_projects/dinov3-testing-stuff/outputs_vitl12_v3/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# =============================================================================
# DINOv3 ViT-Large/16 12-band BigEarthNet Pretraining with Scratch Storage
# =============================================================================
#
# This script uses pre-copied BigEarthNet data from /scratch for faster I/O
# Data must be copied first: bash scripts/copy_bigearthnet_to_scratch.sh
# =============================================================================

echo "Starting DINOv3 ViT-L/16 12-band BigEarthNet pretraining"
echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo ""

# Configuration - using pre-copied data in scratch
SCRATCH_DEST="/scratch/users/$USER/BigEarthNet-S2"

# Verify scratch data exists
if [ ! -d "$SCRATCH_DEST" ]; then
    echo "ERROR: BigEarthNet not found in scratch at $SCRATCH_DEST"
    echo "Please copy it first using: bash scripts/copy_bigearthnet_to_scratch.sh"
    exit 1
fi

echo "Using BigEarthNet from scratch: $SCRATCH_DEST"
PATCH_COUNT=$(find "$SCRATCH_DEST" -type d -name "*_*_*" 2>/dev/null | wc -l)
echo "Patches found: $PATCH_COUNT (expected ~549,000)"

if [ "$PATCH_COUNT" -lt 500000 ]; then
    echo "WARNING: Only $PATCH_COUNT patches found (expected ~549k)"
    echo "Copy may be incomplete. Consider re-copying."
fi

DATA_PATH="$SCRATCH_DEST"
echo "Data source: SCRATCH (fast local SSD)"
echo ""

# Change to repo root
cd /home/carmenoliver/my_projects/dinov3-testing-stuff

# Set PYTHONPATH to use local dinov3-testing-stuff code (not installed package)
export PYTHONPATH=/home/carmenoliver/my_projects/dinov3-testing-stuff:$PYTHONPATH

# Run training
/home/carmenoliver/.conda/envs/envr/bin/python \
  dinov3/train/train.py \
  --config-file dinov3/configs/train/train_vitl16_12b.yaml \
  --output-dir outputs_vitl12_v3 \
  train.dataset_path="BigEarthNet12Band:root=$DATA_PATH:band_masking_prob=0.3" \
  train.batch_size_per_gpu=32\
  optim.lr=0.0005 \
  optim.warmup_epochs=1 \
  optim.epochs=5\
  train.num_workers=4

echo ""
echo "Training finished at: $(date)"
