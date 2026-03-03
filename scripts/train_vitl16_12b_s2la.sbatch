#!/bin/bash
#SBATCH --job-name=dinov3_vitl16_12b_v4
#SBATCH --partition=gpu
#SBATCH --output=/home/carmenoliver/my_projects/dinov3-testing-stuff/outputs_vitl12_v4/slurm-%j.out
#SBATCH --error=/home/carmenoliver/my_projects/dinov3-testing-stuff/outputs_vitl12_v4/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# =============================================================================
# DINOv3 ViT-Large/16 12-band Core-S2L2A Pretraining (v4)
# =============================================================================

echo "Starting DINOv3 ViT-L/16 12-band Core-S2L2A pretraining (v4)"
echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo ""

# Change to repo root
cd /home/carmenoliver/my_projects/dinov3-testing-stuff

# Set PYTHONPATH to use local dinov3-testing-stuff code (not installed package)
export PYTHONPATH=/home/carmenoliver/my_projects/dinov3-testing-stuff:$PYTHONPATH

# Path to your python executable in the conda env
PYTHON_BIN=/home/carmenoliver/.conda/envs/envr/bin/python

# Output directory
OUTDIR=outputs_vitl12_v4
mkdir -p ${OUTDIR}

#Add band masking with 30% probability
# Run training (override compute_precision.param_dtype=fp16 to enable AMP fp16)
${PYTHON_BIN} dinov3/train/train.py \
  --config-file dinov3/configs/train/train_vitl16_12b.yaml \
  --output-dir ${OUTDIR} \
  train.dataset_path="CoreS2L2A:root=/data/databases/Core-S2L2A/images:band_masking_prob=0.3" \
  train.batch_size_per_gpu=32 \
  train.num_workers=8 \
  optim.lr=0.0001 \
  optim.warmup_epochs=2 \
  optim.epochs=5 \
  compute_precision.param_dtype=fp16
  
echo ""
echo "Training finished at: $(date)"
