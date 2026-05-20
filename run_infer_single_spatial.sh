#!/bin/bash
#SBATCH --job-name=arkya_v4_infer_spatial
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/logs/infer_spatial_%j.log

eval "$(conda shell.bash hook)"
conda activate accident_comp

BASE_DIR="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4"

# Change this variable to test different videos
TEST_VIDEO="/ihub/homedirs/sc_hrrs/arkya/videos/_Etcfb7cUH8_00.mp4"

mkdir -p logs
cd $BASE_DIR

# Run the single-video spatial attention script
srun python -u infer_single_spatial.py \
  --video_path "$TEST_VIDEO"
