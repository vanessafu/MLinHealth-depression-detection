#!/usr/bin/env bash
# 试跑：dev 集的 session 上比较三种转写来源，一直跑到 Stage 1（5 折 OOF）结束。
#   1 two_track   原本的管线
#   2 official    Ellie 问题取自新方法，受访者回答换成官方 transcript
#   3 compressed  新方法
# 每一步都有缓存，中断后重跑同一条命令会跳过已完成的部分。
#   bash scripts/analysis/run_trial_stage1.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

python scripts/analysis/trial_setup.py
SIDS=$(python -c "import pandas as pd; print(' '.join(map(str, sorted(pd.read_csv('data/trial/labels/train_split.csv').Participant_ID))))")
echo "sessions: $SIDS"

echo "===== preprocessing: compressed"
python scripts/preprocessing/run_preprocess.py --config config/trial/compressed.yaml --sids $SIDS --steps vad transcribe qa
echo "===== preprocessing: two_track"
python scripts/preprocessing/run_preprocess.py --config config/trial/two_track.yaml --sids $SIDS --steps vad transcribe qa
echo "===== preprocessing: official answers"
python scripts/analysis/official_qa.py --config config/trial/official.yaml --sids $SIDS

echo "===== folds（三种共用 data/trial/splits/folds.csv）"
python scripts/preprocessing/prepare_labels.py --config config/trial/base.yaml

echo "===== 检查三种转写的 Q-A 对是否覆盖同一批 session（否则对比不公平）"
missing=0
for s in $SIDS; do
  for d in data/interim data/interim_two_track data/interim_official; do
    [[ -f $d/qa_pairs/norapport/$s.json ]] || { echo "缺 $d/qa_pairs/norapport/$s.json"; missing=1; }
  done
done
[[ $missing == 0 ]] || { echo "有 session 缺 Q-A 对，先补齐再跑 Stage 1"; exit 1; }

# 显存碎片：fold 1 OOM 时有 2.7 GB 被预留但用不上
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set +e                                    # 某一折失败不影响其他折 / 其他转写
for v in two_track official compressed; do
  out=outputs/trial_$v/stage1
  if [[ -f $out/teacher_oof.npz ]]; then echo "===== stage 1: $v 已完成，跳过"; continue; fi
  echo "===== stage 1: $v"
  [[ -f $out/retrieval.json ]] || python scripts/stage1-text/retrieve.py --config config/trial/$v.yaml
  for f in 0 1 2 3 4; do
    for attempt in 1 2; do
      if [[ -f $out/fold$f/probs.npz ]]; then echo "fold $f 已完成"; break; fi
      python scripts/stage1-text/train_stage1.py --config config/trial/$v.yaml --folds $f \
        || echo "!!!!! stage 1 $v fold $f 第 $attempt 次失败"
    done
  done
  python scripts/stage1-text/train_stage1.py --config config/trial/$v.yaml --assemble-only
done
set -e

echo "===== compare"
python scripts/analysis/compare_trial.py
