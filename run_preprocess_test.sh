#!/bin/bash
#SBATCH --job-name=arkya_v4_preproc_test
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/logs/preprocess_test_%j.log

eval "$(conda shell.bash hook)"
conda activate accident_comp

BASE_DIR="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4"
TEST_CSV="/ihub/homedirs/sc_hrrs/arkya/test_metadata.csv"
VIDEO_DIR="/ihub/homedirs/sc_hrrs/arkya"
OUTPUT_DIR="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/test_clips"

mkdir -p logs
cd $BASE_DIR

srun python -u preprocess_test_clips.py \
  --test_csv "$TEST_CSV" \
  --video_dir "$VIDEO_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --output_csv "${OUTPUT_DIR}/test_clips_metadata.csv"
