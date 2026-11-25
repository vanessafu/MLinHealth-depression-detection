#!/bin/bash -l
# bench2 的单个 Stage 1 任务（数组）。已有结果的任务会直接跳过，失败后重交同一数组即可。
#   sbatch --array=0-39 --time=02:00:00 --job-name=b2-short scripts/evaluation/bench2_run.sbatch short
#   sbatch --array=0-39 --time=04:30:00 --job-name=b2-long  scripts/evaluation/bench2_run.sbatch long
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --output=/home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection/logs/bench2/%x-%A_%a.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail
cd /home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection
source scripts/env.sh
python scripts/evaluation/bench2.py run --group "$1" --work "$TMPDIR"
