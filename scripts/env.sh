# FAU NHR Alex 集群环境。所有 venv / 缓存 / 模型都放在本项目目录下（.venv、.cache），不碰其他项目。
#   source scripts/env.sh          # 登录节点、交互作业或 sbatch 脚本里都可以
#   bash scripts/setup_env.sh      # 第一次：在登录节点上建 venv

EDAIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EDAIC_ROOT

module purge
module load uv/0.11.4

export UV_CACHE_DIR="$EDAIC_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$EDAIC_ROOT/.cache/python"
export UV_PROJECT_ENVIRONMENT="$EDAIC_ROOT/.venv"
export HF_HOME="$EDAIC_ROOT/.cache/huggingface"          # whisper / wav2vec2 对齐 / pyannote 模型
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"    # 复用已有的 HF token
export TORCH_HOME="$EDAIC_ROOT/.cache/torch"
export XDG_CACHE_HOME="$EDAIC_ROOT/.cache/xdg"
export NLTK_DATA="$EDAIC_ROOT/.cache/nltk"                # whisperx 对齐分句用 punkt_tab
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$NLTK_DATA"

# 计算节点不能直接上网，下载走 NHR 代理
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export http_proxy=http://proxy.nhr.fau.de:80
  export https_proxy=http://proxy.nhr.fau.de:80
fi

if [[ -f "$EDAIC_ROOT/.venv/bin/activate" ]]; then
  source "$EDAIC_ROOT/.venv/bin/activate"
else
  echo "env.sh: $EDAIC_ROOT/.venv 不存在，先在登录节点运行 bash scripts/setup_env.sh" >&2
fi
