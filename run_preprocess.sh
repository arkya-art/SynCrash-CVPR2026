#!/bin/bash
#SBATCH --job-name=arkya_v4_prep
#SBATCH --partition=1gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4/logs/preprocess_v4_%j.log

eval "$(conda shell.bash hook)"
conda activate accident_comp

BASE_DIR="/ihub/homedirs/sc_hrrs/arkya/accident_cvpr_arkya_v4"
cd $BASE_DIR
mkdir -p logs

srun python -u preprocess_clips.py
