#!/bin/bash -l
# before（当前 WhisperX 流程）+ after（官方转写修复）在所有 session 上重跑，按 Slurm 数组切分：
#   sbatch --array=0-9 scripts/preprocessing/run_repair_array.sbatch
# 每个任务跑 data/interim/repair_sids.txt 里第 i 份（共 SLURM_ARRAY_TASK_COUNT 份）
#SBATCH --job-name=edaic-repair
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection/logs/%x-%A_%a.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail
cd /home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection
source scripts/env.sh
python scripts/preprocessing/run_preprocess.py --sid-file data/interim/repair_sids.txt \
    --chunk "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" --steps vad transcribe repair qa --overwrite \
    --log "data/interim/preprocess_log_${SLURM_ARRAY_TASK_ID}.json"
