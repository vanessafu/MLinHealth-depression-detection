#!/bin/bash -l
# dev 集端到端零样本测试：预处理 (vad/transcribe/qa) -> Stage 1 检索 + LLM 逐题打分 -> 和 PHQ-8 标签比对
#   sbatch scripts/evaluation/dev_zeroshot.sbatch
#   sbatch --export=NONE,MODEL=Qwen/Qwen3-8B scripts/evaluation/dev_zeroshot.sbatch   # 换模型（预处理有缓存，几分钟）
# 换 A100 (40GB)：sbatch --partition=a100 --gres=gpu:a100:1 scripts/evaluation/dev_zeroshot.sbatch
#SBATCH --job-name=dev-zeroshot
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=05:00:00
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
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
EXP_NAME=${EXP_NAME:-dev_zs_$(basename "$MODEL" | tr 'A-Z.-' 'a-z__')}
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/evaluation/dev_zeroshot.py --set stage1.model.base="$MODEL" experiment.name="$EXP_NAME" "$@"
