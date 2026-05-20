#!/bin/bash
#SBATCH --job-name=arkya_v4_infer
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/logs/inference_v4_%j.log

eval "$(conda shell.bash hook)"
conda activate accident_comp

BASE_DIR="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4"
TEST_CSV="/ihub/homedirs/sc_hrrs/arkya/test_metadata.csv"
VIDEO_DIR="/ihub/homedirs/sc_hrrs/arkya"
MODEL_WEIGHTS="${BASE_DIR}/checkpoints/best_binary_model.pth"
OUTPUT_CSV="${BASE_DIR}/submission_v4.csv"

mkdir -p logs
cd $BASE_DIR

srun python -u inference_binary.py \
  --test_csv "$TEST_CSV" \
  --video_dir "$VIDEO_DIR" \
  --model_weights "$MODEL_WEIGHTS" \
  --out_csv "$OUTPUT_CSV" \

