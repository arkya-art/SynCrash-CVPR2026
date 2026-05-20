#!/bin/bash
#SBATCH --job-name=arkya_v4_train
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/logs/train_v4_%j.log

eval "$(conda shell.bash hook)"
conda activate accident_comp

BASE_DIR="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints"
CLIPS_CSV="/ihub/homedirs/sc_hrrs/arkya/sim_dataset/binary_clips_metadata.csv"
CLIPS_DIR="/ihub/homedirs/sc_hrrs/arkya/sim_dataset"

mkdir -p $CHECKPOINT_DIR
mkdir -p logs
cd $BASE_DIR

srun python -u train_binary.py \
  --clips_csv "$CLIPS_CSV" \
  --clips_dir "$CLIPS_DIR" \
  --output_dir "$CHECKPOINT_DIR" \
  --batch_size 2 \
  --grad_accum 16 \
  --epochs 10 \
  --lr 2e-5 \
  --samples_per_epoch 6000
