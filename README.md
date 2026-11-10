# MLinHealth-depression-detection

E-DAIC PHQ-8 预测：只用访谈录音 `XXX_AUDIO.wav`，Stage 1 用 LLM + LoRA 逐题打分（文本来自 WhisperX 转写），Stage 2 用冻结的 WavLM 特征 + 注意力 MIL 学习文本预测的残差，两阶段用 5 折 OOF 连接，最后按 β 融合。

- 超参数和“改了要重跑什么”：[config/README.md](config/README.md)
- 设计细节：《E-DAIC PHQ-8 预测管线手册》

## 目录结构

```text
config/
  default.yaml            全部超参数（推荐起点）
  eval_baseline.yaml      基线 A0 / A1
  ablations/              消融 A4 A5 A6 A8 A9（只写和 default 不同的地方）
  README.md               超参数说明（中文）
data/                     不进 git，只保留目录
  raw/                    ← 原始录音 XXX_AUDIO.wav（或 XXX_P/XXX_AUDIO.wav）
  labels/                 ← train/dev/test_split.csv、Detailed_PHQ8_Labels.csv
  interim/                VAD、转写、Q-A 对、片段表（自动生成）
  features/               WavLM 特征缓存（自动生成）
  splits/                 固定的 5 折划分 folds.csv（自动生成）
resources/
  ellie_questions.csv     Ellie 问题库 + 问题→题号映射（需要人工完善）
  item_queries.yaml       PHQ-8 每题定义和检索查询
scripts/                  命令行入口
  preprocessing/          prepare_labels / run_preprocess / qc_report
  stage1-text/            retrieve / train_stage1
  stage2-audio/           precheck / train_stage2
  evaluation/             evaluate / run_baselines
src/                      代码（本身就是 Python 包，import src.xxx）
  preprocessing/          vad, transcribe, ellie, qa_pairs, segments, wavlm_features, pipeline
  stage1/                 retrieval, prompts, dataset, model, losses, train, teacher
  stage2/                 dataset, model, losses, train
  evaluation/             metrics, report
  fusion.py  splits.py  labels.py  config.py
tests/                    合成数据上的冒烟测试（不需要音频和 GPU）
outputs/<实验名>/          stage1/ stage2/ final/（不进 git）
```

## 环境

```bash
# 本机（只装核心依赖，能跑测试和 Stage 2）
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# GPU 机器（Linux + CUDA）
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[all]"          # = preprocess + stage1 + dev
export HF_TOKEN=hf_xxx              # pyannote 说话人分离需要；还要在 HF 上接受 pyannote 模型的使用条款
```

依赖分组见 [pyproject.toml](pyproject.toml)：`preprocess`（silero-vad、whisperx、WavLM）、`stage1`（transformers、peft、sentence-transformers、bitsandbytes）。

## 运行顺序

```bash
# 0. 标签检查 + 固定 5 折划分（只需要 data/labels/，不需要音频）
python scripts/preprocessing/prepare_labels.py

# 1. 预处理（每步有缓存，可以中断后继续）
python scripts/preprocessing/run_preprocess.py
python scripts/preprocessing/qc_report.py          # 人工核对 Ellie 识别和 Q-A 切分，补全问题库

# 2. Stage 1
python scripts/stage1-text/retrieve.py             # 看打印的 max r 分布，校准 evidence_threshold
python scripts/stage1-text/train_stage1.py         # 5 折 → outputs/main/stage1/teacher_oof.npz

# 3. Stage 2
python scripts/stage2-audio/precheck.py            # 两个前提检查
python scripts/stage2-audio/train_stage2.py --grid # 27 组超参数，按 CV CCC 排序
python scripts/stage2-audio/train_stage2.py --use-best-grid

# 4. 融合与评估
python scripts/evaluation/evaluate.py --text-only  # A3：只用文本
python scripts/evaluation/evaluate.py              # A7：完整管线
```

基线和消融：

```bash
# A0 + A1
python scripts/stage1-text/retrieve.py     --config config/eval_baseline.yaml
python scripts/stage1-text/train_stage1.py --config config/eval_baseline.yaml --zero-shot
python scripts/evaluation/run_baselines.py --config config/eval_baseline.yaml

# 只涉及 Stage 2 的消融（A6 / A9）直接复用主实验的老师预测
python scripts/stage2-audio/train_stage2.py --config config/ablations/A6_no_aux.yaml
python scripts/evaluation/evaluate.py       --config config/ablations/A6_no_aux.yaml

# A8 先在主实验里生成非 OOF 老师预测
python scripts/stage1-text/train_stage1.py --full-fit
```

## 测试

```bash
python -m pytest -q
```
