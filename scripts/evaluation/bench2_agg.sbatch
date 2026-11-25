#!/bin/bash -l
# bench2 汇总：老师、dev 指标、bootstrap、修正版 Stage 2 + β 融合 -> logs/bench2/results.json
#SBATCH --job-name=b2-agg
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection/logs/bench2/%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail
cd /home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection
source scripts/env.sh
python scripts/evaluation/bench2.py aggregate --work "$TMPDIR" "$@"
