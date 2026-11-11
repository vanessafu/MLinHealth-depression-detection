#!/usr/bin/env bash
# 在登录节点上（有网）建 .venv 并安装预处理依赖（silero-vad + whisperx）。
#   bash scripts/setup_env.sh            # 默认 extras: preprocess,dev
#   EXTRAS=all bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh || true
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[${EXTRAS:-preprocess,dev}]"
python -c "import torch, whisperx, silero_vad; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import nltk; nltk.download('punkt_tab', download_dir='$NLTK_DATA')"
