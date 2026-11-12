# 配置与超参数说明

所有超参数都在 [default.yaml](default.yaml) 里，已经按《E-DAIC PHQ-8 预测管线手册》填好推荐起点。本文说明每一项能不能改、往哪个方向改，以及**改完之后要重跑哪一步**（预处理和检索都有缓存，改错位置的话新设置不会生效）。

## 怎么改

三种方式，优先级从低到高：

1. **直接改 `default.yaml`**：改的是全局默认值。
2. **新建一个继承文件**：只写和默认值不同的地方。消融实验都用这种方式，见 [ablations/](ablations/)。

   ```yaml
   _base_: default.yaml        # 路径相对于当前文件
   experiment: {name: my_exp}  # 产物写到 outputs/my_exp/，不会覆盖其他实验
   stage2:
     loss: {lambda_res: 0.1}
   ```

3. **命令行临时覆盖**：`--set 键=值`，值按 YAML 解析（数字、列表、`null`、`true` 都行）。

   ```bash
   python scripts/stage2-audio/train_stage2.py --set stage2.loss.w_mov=0 stage2.train.seeds=[0,1]
   ```

每次训练都会把实际生效的完整配置存到 `outputs/<实验名>/<阶段>/config.yaml`，方便事后复现。

## 图例

| 标记 | 含义 |
| --- | --- |
| 🖥️ **环境** | 和机器、显存有关，按你的 GPU 改，不影响结论 |
| 📏 **校准** | 拿到数据后必须人工抽查再定，默认值只是占位 |
| 🎯 **调参** | 计划内要搜索的超参数，只能用 train 的 5 折 CV 选，不能看 dev |
| 🧪 **消融** | 消融实验用的开关，主实验保持默认 |
| 🔒 **固定** | 手册里定为固定值，没有充分理由不要改 |

---

## 1. 实验与路径

| 键 | 默认值 | 类别 | 说明 |
| --- | --- | --- | --- |
| `experiment.name` | `main` | — | 产物目录 `outputs/<name>/`。每个实验起一个新名字 |
| `experiment.seed` | 42 | 🔒 | Stage 1 的随机种子（第 f 折实际使用 seed + f） |
| `experiment.device` | `cuda` | 🖥️ | `cuda` / `cpu` / `mps` |
| `paths.raw_audio` | `data/raw` | 🖥️ | 放 `XXX_AUDIO.wav`。官方解压出来的 `XXX_P/XXX_AUDIO.wav` 结构也能识别 |
| `paths.labels_dir` | `data/labels` | 🖥️ | 放 `train_split.csv`、`dev_split.csv`、`test_split.csv`、`Detailed_PHQ8_Labels.csv` |
| `paths.egemaps_dir` | `null` | — | 可选。只有用官方 eGeMAPS 做 Stage 2 前提检查时才需要 |
| `labels.*` | E-DAIC 官方列名 | — | 你的 CSV 列名不同时改这里。`item_cols` 的顺序就是题号 1–8 |
| `labels.ai_id_range` | `[600, 718]` | 🔒 | 这个范围内的 ID 算 AI 采集，其余算 WoZ |

## 2. K 折划分

| 键 | 默认值 | 类别 | 说明 |
| --- | --- | --- | --- |
| `splits.n_folds` | 5 | 🔒 | 两个阶段共用 |
| `splits.phq_bin_edges` | `[0,5,10,15,20,25]` | 🔒 | 分层用的 PHQ 分箱 |
| `splits.stratify_by` | `[phq_bin, gender, mode]` | — | 某一层人数少于折数时，会从后往前自动去掉变量，实际用到的变量会打印出来 |
| `splits.seed` | 42 | 🔒 | |

> ⚠️ `data/splits/folds.csv` 生成后**不要再改**。`prepare_labels.py` 默认不覆盖已有文件，需要重新生成时加 `--force`，然后两个阶段都要重跑。

## 3. 预处理

**改完要重跑**：`scripts/preprocessing/run_preprocess.py --steps <从被改的那一步开始往后的所有步骤> --overwrite`

| 键 | 默认值 | 类别 | 范围 / 建议 | 从哪一步重跑 |
| --- | --- | --- | --- | --- |
| `transcribe.method` | `compressed` | 🧪 | **切回原版：`--config config/two_track.yaml`**（只改方法和 interim 目录，缓存不混；在同一目录里换方法会报错，防止误用另一种方法的转写）。`compressed`：逐帧压缩（不降噪）+ 无 VAD 主转写，WhisperX 补漏，声纹判说话人；`two_track`：旧的 Silero 分轨 + 融合（下面 `vad.thresholds` / `ellie_track.*` / `speaker.*` 只对它生效，`speaker.phrases` / `speaker.opening_min_score` 除外：两种方法都用）。5 个 session 上 compressed：官方转写召回 0.967 vs 0.958，Ellie 台词命中 43.6 vs 39.0（`src/preprocessing/compress.py` / `asr_merge.py` / `voiceprint.py`；对比实验脚本已删，见 git 历史） | transcribe |
| `transcribe.chunk_size` | 8 | 📏 | raw 路 WhisperX 的块长（秒）。30 时轻声段对不上会把整块时间戳带偏 3 s 以上 | transcribe |
| `transcribe.compress.*` | -24 / 30 / 3 dB | 📏 | 目标响度 / 最大增益 / 门限。门限去掉会在放大的底噪上出循环幻觉 | transcribe |
| `transcribe.speaker.*` | 见 default.yaml | 📏 | 声纹（wespeaker）kNN + 台词 + 响度判说话人；`embed_audio` 选声纹从原始还是压缩后音频上取 | transcribe |
| `vad.thresholds` | [0.12, 0.7] | 📏 | 低阈值为主、高阈值补漏，两次分轨转写融合（单一阈值在不同录音上最优值相反）。只用于 `two_track` | vad, transcribe |
| `vad.min_speech_ms` / `min_silence_ms` / `speech_pad_ms` | 250 / 300 / 100 | 🔒 | | vad |
| `whisperx.model` | `large-v3` | 🔒 | 显存紧张可用 `large-v3-turbo` | transcribe |
| `whisperx.compute_type` | `float16` | 🖥️ | GPU 用 `float16`，CPU 用 `int8` | transcribe |
| `whisperx.batch_size` | 16 | 🖥️ | 显存不够就调小，不影响结果 | transcribe |
| `ellie_track.*` | 见 default.yaml | 📏 | quiet 轨：VAD 区间挖掉后降噪 + 逐段 RMS 标准化（录音里一个人常比另一个小 7–30 dB） | transcribe |
| `speaker.*` | 见 default.yaml | 📏 | 逐句判别 Ellie / 受访者：文本 / 响度（按 d' 自适应）/ 轨道 / 问句 加权，只对 `two_track` 生效（调参脚本 `tune_speaker.py` 已删，见 git 历史）。例外：`phrases`（台词库）和 `opening_min_score`（访谈开始）两种方法都用 | transcribe |
| `whisperx.utterance_gap_sec` | 1.0 | 📏 | 同一个人两个词之间超过这个间隔就切成两句话 | qa |
| `ellie.question_bank` | `resources/ellie_questions.csv` | 📏 | **需要你补全**，见下文“资源文件” | qa |
| `ellie.match_threshold` | 0.80 | 📏 | 0.75–0.90。识别 Ellie 或分类她的话时漏得多就调低，误匹配多就调高 | qa |
| `ellie.unmatched_as` | `followup` | 📏 | Ellie 说了但问题库里没有的句子：`followup`（并入当前问题）或 `backchannel`（忽略） | qa |
| `qa.keep_rapport` | `false` | 🧪 | 消融 A5。产物写到单独的 `rapport/` 子目录，不会覆盖默认结果 | qa |
| `qa.trim_opening` / `trim_closing` | `true` | 🔒 | 掐头去尾 | qa |
| `segments.min_sec` | 1.0 | 🔒 | 回答短于 1 秒不切音频片段（文本仍然保留） | segments |
| `segments.max_sec` | 20.0 | 🔒 | | segments |
| `segments.margin_sec` | 0.2 | 🔒 | 两端余量，不会延伸进 Ellie 的语音 | segments |
| `segments.split_pause_sec` | 0.3 | 🔒 | 切长片段时优先在 ≥ 0.3 秒的停顿处切 | segments |
| `wavlm.layers` | `all` | 🖥️ | 磁盘不够时改成层号列表，如 `[0, 6, 12, 18, 24]`。Stage 2 会自动适配层数 | wavlm |
| `wavlm.dtype` | `float16` | 🔒 | | wavlm |

## 4. Stage 1：文本逐题模型

### 检索

**改完要重跑**：`scripts/stage1-text/retrieve.py`，然后重训 Stage 1。

| 键 | 默认值 | 类别 | 说明 |
| --- | --- | --- | --- |
| `retrieval.encoder` | `BAAI/bge-large-en-v1.5` | 🔒 | |
| `retrieval.item_queries` | `resources/item_queries.yaml` | 📏 | 每道题的检索查询，可以增删 |
| `retrieval.use_question_map` | `true` | 🧪 | 消融 A4 |
| `retrieval.top_n` | 6 | 🔒 | 推理时每题的候选数 N |
| `retrieval.store_top` | 8 | — | 保存的候选数，必须 ≥ `augment.n_max` |
| `retrieval.evidence_threshold` | 0.60 | 📏 | **必须校准**。`retrieve.py` 会打印各题 max r 的分位数和 ev=1 的比例。抽几个 session 看看“确实没聊到这道题”时 max r 是多少，阈值定在它上面一点 |

### 模型

| 键 | 默认值 | 类别 | 说明 |
| --- | --- | --- | --- |
| `model.base` | `Qwen/Qwen2.5-7B-Instruct` | 🎯 | 备选 `meta-llama/Llama-3.1-8B-Instruct`（需要在 HF 上申请访问） |
| `model.dtype` | `bfloat16` | 🖥️ | 不支持 bf16 的卡（V100、T4）改成 `float16` |
| `model.load_in_4bit` | `false` | 🖥️ | 24GB 显存建议打开（QLoRA）。只在 Linux 上可用 |
| `model.attn_implementation` | `sdpa` | 🖥️ | 装了 flash-attn 可以改成 `flash_attention_2` |
| `model.max_length` | 2048 | 🖥️ | 超出时从左边截断（会截掉提示词开头），所以最好先靠 `max_answer_words` 控制长度 |
| `model.max_answer_words` | 150 | 📏 | 每个回答在提示词里最多保留多少词。如果大量提示词超过 2048 token，就调小 |
| `model.gradient_checkpointing` | `true` | 🖥️ | 显存充足时关掉可以提速 |
| `model.use_chat_template` | `true` | 🔒 | 用 Instruct 模型时保持打开 |
| `model.lora.r` / `alpha` / `dropout` | 16 / 32 / 0.05 | 🔒 | |
| `model.lora.target_modules` | `[q_proj, k_proj, v_proj, o_proj]` | 🔒 | Qwen 和 Llama 的模块名相同 |

### Loss

| 键 | 默认值 | 类别 | 范围 | 说明 |
| --- | --- | --- | --- | --- |
| `loss.mu` | 0.5 | 🎯 | 0.25–1.0 | 期望值平方误差的权重。越大越接近回归，越小越接近纯分类 |
| `loss.lambda_sum` | 0.5 | 🎯 | 0.25–1.0 | session 级总分项的权重 |
| `loss.class_weight` | `inv_sqrt_freq` | 🔒 | `inv_sqrt_freq` / `none` | w_k(c) ∝ 1/√频率，按训练样本加权后均值归一化为 1 |
| `loss.class_weight_smoothing` | 1.0 | 🔒 | | 频数加 1，防止某一类在训练折里没有样本 |

### 训练

| 键 | 默认值 | 类别 | 说明 |
| --- | --- | --- | --- |
| `train.sessions_per_step` | 2 | 🔒 | 每个优化步 2 个 session × 8 题 = 16 条序列 |
| `train.micro_batch_sessions` | 1 | 🖥️ | 每次前向的 session 数，其余靠梯度累积。**最小单位是 1 个 session（8 条序列）**，因为 L_sum 需要同一个 session 的 8 道题 |
| `train.lr` | 1e-4 | 🎯 | 5e-5 – 2e-4 |
| `train.warmup_ratio` | 0.05 | 🔒 | 线性预热，之后线性衰减 |
| `train.max_epochs` / `patience` | 5 / 2 | 🔒 | 按本折验证集上 ŷ^text 的 MAE 早停 |
| `train.max_grad_norm` | 1.0 | 🔒 | |
| `train.eval_batch_sessions` | 4 | 🖥️ | 推理时一次几个 session，只影响速度 |
| `augment.n_min` / `n_max` | 4 / 8 | 🔒 | 训练时 N 随机取值的范围 |
| `augment.drop_one_prob` | 0.3 | 🔒 | 随机丢掉 1 个候选的概率 |
| `augment.shuffle_prob` | 0.2 | 🔒 | 打乱候选顺序的概率（其余情况按访谈时间排序） |

## 5. Stage 2：音频残差模型

改了 Stage 2 的参数**只需要重跑 Stage 2**，WavLM 特征和 Stage 1 老师预测都不用动。

| 键 | 默认值 | 类别 | 说明 |
| --- | --- | --- | --- |
| `teacher.kind` | `oof` | 🧪 | `oof` / `in_sample`（消融 A8）/ `zeroshot` |
| `teacher.stage1_dir` | `null` | 🧪 | 消融时指向 `outputs/main/stage1`，复用主实验的老师预测，不用重训 Stage 1 |
| `model.proj_dim` / `attn_dim` / `hidden_dim` | 256 / 128 / 128 | 🔒 | |
| `model.dropout` | 0.3 | 🔒 | |
| `model.max_delta` | 6.0 | 🔒 | 总分最大修正 ±6 分 |
| `model.pooling` | `attention` | 🧪 | `mean` 即消融 A9 |
| `model.use_pos` | `true` | 🔒 | 注意力打分时是否拼上片段位置 |
| `loss.w_tired` | 0.5 | 🎯 | 网格 {0, 0.25, 0.5} |
| `loss.w_mov` | 0.5 | 🎯 | 网格 {0, 0.25, 0.5}。`precheck.py` 报告 moving 正样本 < 25 时固定为 0 |
| `loss.lambda_res` | 0.05 | 🎯 | 网格 {0.01, 0.05, 0.1} |
| `loss.ccc_weight` | 0.5 | 🔒 | |
| `loss.huber_delta` | 0.1 | 🔒 | 在归一化尺度（/24）上，约 2.4 分 |
| `loss.use_lds` / `lds_sigma` | `false` / 2.0 | — | LDS 样本权重，可选实验 |
| `train.lr` / `weight_decay` | 1e-3 / 1e-2 | 🔒 | |
| `train.max_epochs` / `patience` | 300 / 30 | 🔒 | 按本折验证集的总分 CCC 早停 |
| `train.max_segments` | 64 | 🔒 | 训练时每个 session 每个 epoch 最多随机抽 64 个片段；推理时用全部片段 |
| `train.batch_sessions` | `null` | 🖥️ | `null` = 整个训练折一个 batch（约 130 个 session，显存约 2–4GB）。显存不够时设成 32 之类，但 batch 内 CCC 的估计会变得不稳 |
| `train.seeds` | `[0,1,2,3,4]` | 🔒 | 5 折 × 5 seed = 25 个模型。调试时可以临时改成 `[0]` |
| `grid.*` | 见上 | 🎯 | `train_stage2.py --grid` 搜索的范围，共 27 组 |

## 6. 融合与评估

| 键 | 默认值 | 类别 | 说明 |
| --- | --- | --- | --- |
| `fusion.beta_grid` | 0, 0.1, …, 1.0 | 🔒 | 在 train OOF 上按 CCC 选 β，不通过梯度学习 |
| `evaluation.severity_bin_edges` | `[0,5,10,15,20,25]` | 🔒 | 分箱 MAE |
| `evaluation.report_ai_subset` | `true` | 🔒 | dev 上额外报告 AI 子集（600–718） |

---

## 资源文件（需要你人工完善）

### `resources/ellie_questions.csv`：Ellie 问题库 + 问题映射表

| 列 | 含义 |
| --- | --- |
| `text` | Ellie 的原话（大小写、标点无所谓，匹配前会统一规范化） |
| `type` | `main`（主问题）/ `followup`（追问）/ `backchannel`（附和）/ `opening`（开场）/ `closing`（告别） |
| `rapport` | 寒暄题填 1，由 `keep_rapport` 决定是否保留 |
| `items` | 这个问题对应的 PHQ 题号，多个用 `;` 分隔，如 `3;4`。命中时 r_ijk = 1 |
| `notes` | 备注 |

现在这份是**起步版本**（约 70 句，凭对 DAIC-WOZ 脚本的了解整理），`items` 列是草拟的映射。拿到数据后：

1. 先跑一遍预处理，再跑 `qc_report.py`：它会列出问题库里**没匹配上**的 Ellie 句子（按出现的 session 数排序），把常见的补进问题库。每个 session 的未匹配句子也存在 `data/interim/qa_pairs/*/<sid>.json` 的 `unmatched_ellie` 字段里。
2. 逐条核对 `items` 映射。
3. **600 段（AI 采集）的脚本可能和 WoZ 不同**，开场和告别的锚点要在 600 段上单独检查。
4. 改完问题库后只需从 `qa` 步骤开始重跑：`run_preprocess.py --steps qa segments wavlm --overwrite`。

### `resources/item_queries.yaml`：每道题的定义和检索查询

`text` 会写进提示词，`queries` 用于 embedding 检索。改完后重跑 `retrieve.py` 和 Stage 1。

---

## 改了什么，要重跑什么（速查）

| 改动 | 预处理 | retrieve | Stage 1 | Stage 2 | evaluate |
| --- | :-: | :-: | :-: | :-: | :-: |
| VAD / WhisperX 参数 | ✅ 从 vad 或 transcribe 开始 | ✅ | ✅ | ✅ | ✅ |
| 问题库、`match_threshold`、`keep_rapport` | ✅ 从 qa 开始 | ✅ | ✅ | ✅ | ✅ |
| 片段规则 | ✅ 从 segments 开始 | | | ✅ | ✅ |
| `wavlm.layers` | ✅ 只跑 wavlm | | | ✅ | ✅ |
| 检索参数、`item_queries.yaml` | | ✅ | ✅ | ✅ | ✅ |
| Stage 1 模型 / loss / 训练参数 | | | ✅ | ✅ | ✅ |
| Stage 2 任意参数 | | | | ✅ | ✅ |
| `fusion.beta_grid` | | | | | ✅ |
