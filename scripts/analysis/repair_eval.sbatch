#!/bin/bash -l
# 转写修复评估（指标 / 准确率 / 对照表 / UMAP），在 run_repair_array.sbatch 之后跑：
#   sbatch --dependency=afterany:<array_job_id> scripts/analysis/repair_eval.sbatch
#SBATCH --job-name=edaic-repair-eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection/logs/%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail
cd /home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection
source scripts/env.sh
python scripts/analysis/repair_eval.py "$@"
