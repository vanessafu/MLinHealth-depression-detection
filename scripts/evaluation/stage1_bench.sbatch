#!/bin/bash -l
# Stage 1 dev benchmark（旧 / 挖掘问题库 × 检索模型；候选相关性 + 零样本 + LoRA）。中间数据都在 $TMPDIR，作业结束即删。
#   sbatch scripts/evaluation/stage1_bench.sbatch
#SBATCH --job-name=stage1-bench
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection/logs/%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail
cd /home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection
source scripts/env.sh
python scripts/evaluation/stage1_bench.py --work "$TMPDIR" --out logs/stage1_bench.json "$@"
