"""Baseline S1-B1 / A2: MentalBERT encodes C_ik, 8 heads x 4 classes, same loss as Stage 1.

TODO: 尚未实现。建议做法：
  - 编码器 mental/mental-bert-base-uncased（需要在 HF 上申请访问），输入 = 第 k 题定义 [SEP] C_ik 拼接文本
  - 取 [CLS]，接 Linear(768, 4) × 8（每题一个头），loss 直接复用 stage1.losses.stage1_loss
  - 训练/OOF 流程复用 stage1.train 的结构（run_fold / assemble_oof），输出 teacher_mentalbert.npz
"""

raise NotImplementedError("MentalBERT baseline is not implemented yet; see module docstring.")
