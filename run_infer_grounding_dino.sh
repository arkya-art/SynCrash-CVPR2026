#!/bin/bash
#SBATCH --job-name=arkya_grounding_dino
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/logs/infer_dino_%j.log

eval "$(conda shell.bash hook)"
conda activate accident_comp

BASE_DIR="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4"

cd $BASE_DIR

# Run the multi-video batched spatial script over submission_v4.csv
srun python -u infer_grounding_dino_batch.py \
  --csv_path "$BASE_DIR/submission_v4.csv" \
  --out_csv "$BASE_DIR/submission_dino.csv"


# # Change this variable to test different videos
# TEST_VIDEO="/ihub/homedirs/sc_hrrs/arkya/videos/__WFqm4i3vE_00.mp4"

# mkdir -p logs
# cd $BASE_DIR

# # Run the single-video spatial attention script
# srun python -u infer_grounding_dino_spatial.py \
#   --video_path "$TEST_VIDEO" \
#   --time_window 2.0

