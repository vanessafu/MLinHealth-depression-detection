#!/bin/bash -l
# 完整管线 / 不检索对照的 dev benchmark。中间数据都在 $TMPDIR，作业结束即删。
#   sbatch --job-name=fb-full  scripts/evaluation/full_bench.sbatch --part full
#   sbatch --job-name=fb-allqa scripts/evaluation/full_bench.sbatch --part allqa
#   sbatch --job-name=fb-official scripts/evaluation/full_bench.sbatch --part official
#SBATCH --job-name=full-bench
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection/logs/%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail
cd /home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection
source scripts/env.sh
python scripts/evaluation/full_bench.py --work "$TMPDIR" "$@"
