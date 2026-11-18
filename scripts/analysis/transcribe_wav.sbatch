#!/bin/bash -l
# 用法：sbatch scripts/analysis/transcribe_wav.sbatch data/interim/tracks/309_thr0.7_combined.wav
#SBATCH --job-name=transcribe-wav
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/b279bb/b279bb39/Abschlussarbeit-multilingual-math-reasoning/MLinHealth-depression-detection/logs/%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail
# 不管在哪个目录提交，都进到项目根目录（sbatch 会把脚本复制到别处，所以从 scontrol 取原始路径）
SCRIPT=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/ Command=/{print $2; exit}' | awk '{print $1}')
ROOT=$(cd "$(dirname "$SCRIPT")/../.." && pwd)
[ -f "$ROOT/scripts/env.sh" ] || ROOT="$SLURM_SUBMIT_DIR"
cd "$ROOT"
source scripts/env.sh
python scripts/analysis/transcribe_wav.py "$@"
