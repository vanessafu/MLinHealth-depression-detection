#!/bin/bash -l
# GPU 上跑预处理。参数原样传给 run_preprocess.py，例如：
#   sbatch scripts/preprocessing/run_preprocess.sbatch --sids 309 --steps transcribe
#SBATCH --job-name=edaic-prep
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/preprocessing/run_preprocess.py "$@"
