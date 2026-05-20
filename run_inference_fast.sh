#!/bin/bash
#SBATCH --job-name=arkya_v4_infer_fast
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/logs/inference_fast_%j.log

eval "$(conda shell.bash hook)"
conda activate accident_comp

BASE_DIR="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4"
CLIPS_CSV="${BASE_DIR}/test_clips/test_clips_metadata.csv"
CLIPS_DIR="${BASE_DIR}/test_clips"
MODEL_WEIGHTS="${BASE_DIR}/checkpoints/best_binary_model.pth"
OUTPUT_CSV="${BASE_DIR}/submission_v4.csv"

mkdir -p logs
cd $BASE_DIR

srun python -u inference_from_clips.py \
  --clips_csv "$CLIPS_CSV" \
  --clips_dir "$CLIPS_DIR" \
  --model_weights "$MODEL_WEIGHTS" \
  --out_csv "$OUTPUT_CSV" \
  --batch_size 8 \
  --num_workers 4
