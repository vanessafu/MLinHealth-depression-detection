"""LLM + LoRA scorer: logits of tokens "0".."3" at the last position (no new output layer)."""
from __future__ import annotations

import torch

from ..config import Cfg


class ItemScorer:
    def __init__(self, cfg: Cfg, device: str, with_lora: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mc = cfg.stage1.model
        self.cfg, self.device = cfg, device
        tok = AutoTokenizer.from_pretrained(mc.base)
        tok.padding_side = "left"        # 最后一个位置就是 "Score:" 之后
        tok.truncation_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tok = tok

        kw = {"torch_dtype": getattr(torch, mc.dtype), "attn_implementation": mc.attn_implementation}
        if mc.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, mc.dtype))
            kw["device_map"] = {"": device}
        model = AutoModelForCausalLM.from_pretrained(mc.base, **kw)
        if not mc.load_in_4bit:
            model = model.to(device)

        if with_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if mc.load_in_4bit:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=mc.gradient_checkpointing)
            lc = mc.lora
            model = get_peft_model(model, LoraConfig(
                r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout,
                target_modules=list(lc.target_modules), task_type="CAUSAL_LM"))
            if mc.gradient_checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
            model.print_trainable_parameters()
        self.model = model

        ids = [tok.encode(t, add_special_tokens=False) for t in mc.score_tokens]
        assert all(len(i) == 1 for i in ids), f"score tokens must be single tokens, got {ids}"
        self.score_ids = torch.tensor([i[0] for i in ids], device=device)

    # ---- text -> model input ----
    def format(self, prompt: str) -> str:
        if self.cfg.stage1.model.use_chat_template and self.tok.chat_template:
            return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                add_generation_prompt=True)
        return prompt + " "

    def encode(self, prompts: list[str]) -> dict:
        texts = [self.format(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.cfg.stage1.model.max_length, add_special_tokens=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    # ---- forward ----
    def _parts(self):
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        return base.get_decoder(), base.get_output_embeddings()

    def score_logits(self, prompts: list[str]) -> torch.Tensor:
        """[len(prompts), 4] logits z_ik. Only the 4 lm_head rows are computed (saves memory)."""
        enc = self.encode(prompts)
        decoder, head = self._parts()
        hidden = decoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        last = hidden[:, -1, :]
        w = head.weight[self.score_ids]                      # [4, H]
        z = last.to(w.dtype) @ w.T
        if getattr(head, "bias", None) is not None:
            z = z + head.bias[self.score_ids]
        return z.float()

    # ---- LoRA state ----
    def lora_state(self) -> dict:
        from peft import get_peft_model_state_dict

        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(self.model).items()}

    def load_lora_state(self, state: dict) -> None:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(self.model, state)
