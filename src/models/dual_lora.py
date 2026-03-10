"""
dual_lora.py - Fast/Slow LoRA wrapper with EMA consolidation.

Architecture (per the SuRe paper):
  - ONE frozen T5-Large backbone
  - TWO LoRA parameter sets: fast (trained) and slow (EMA shadow)
  - Training: base + fast LoRA
  - Inference: base + slow LoRA (weights swapped in temporarily)

Optimized for RTX 4050 6GB:
  - fp16 mixed precision throughout
  - gradient_checkpointing on base model
  - only fast LoRA params require gradients at train time
  - slow LoRA stored as nn.ParameterList (no grad, moves with .to(device))
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import T5ForConditionalGeneration


class DualLoRAModel(nn.Module):
    def __init__(
        self,
        base_model: T5ForConditionalGeneration,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_target_modules: list = None,
        beta: float = 0.995,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules = ["q", "v"]

        # Freeze base weights
        for param in base_model.parameters():
            param.requires_grad_(False)

        if use_gradient_checkpointing:
            base_model.gradient_checkpointing_enable()

        # ONE backbone + fast LoRA adapter (trainable)
        fast_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        self.fast_model = get_peft_model(base_model, fast_cfg)

        # Slow LoRA: shadow parameter tensors only — no second backbone.
        # nn.ParameterList ensures automatic device movement with .to(device).
        slow_params = [
            nn.Parameter(p.detach().clone(), requires_grad=False)
            for _, p in self._lora_params(self.fast_model)
        ]
        self._slow_lora = nn.ParameterList(slow_params)

        self.beta = beta
        # slow is initialised equal to fast (already done by clone above)

    # ------------------------------------------------------------------
    def _lora_params(self, model):
        """Yield (name, param) for LoRA adapter parameters only."""
        for name, param in model.named_parameters():
            if "lora_" in name:
                yield name, param

    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, labels=None, use_slow: bool = False):
        if not use_slow:
            return self.fast_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # Inference with slow LoRA: temporarily swap fast weights → slow,
        # run forward, then restore. No grad context since this is eval-only.
        fast_params = list(self._lora_params(self.fast_model))
        saved = [p.data.clone() for _, p in fast_params]
        try:
            with torch.no_grad():
                for (_, pf), ps in zip(fast_params, self._slow_lora):
                    pf.data.copy_(ps.data)
            output = self.fast_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            with torch.no_grad():
                for (_, pf), sv in zip(fast_params, saved):
                    pf.data.copy_(sv)
        return output

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """CE loss using fast LoRA."""
        outputs = self.forward(**batch, use_slow=False)
        return outputs.loss

    @torch.no_grad()
    def ema_update(self):
        """θ_slow ← β·θ_slow + (1-β)·θ_fast"""
        for (_, pf), ps in zip(self._lora_params(self.fast_model), self._slow_lora):
            ps.data.mul_(self.beta).add_((1.0 - self.beta) * pf.data)

    def generate(self, **kwargs):
        """Generate using slow LoRA weights (weight-swap approach)."""
        fast_params = list(self._lora_params(self.fast_model))
        saved = [p.data.clone() for _, p in fast_params]
        try:
            with torch.no_grad():
                for (_, pf), ps in zip(fast_params, self._slow_lora):
                    pf.data.copy_(ps.data)
            output = self.fast_model.generate(**kwargs)
        finally:
            with torch.no_grad():
                for (_, pf), sv in zip(fast_params, saved):
                    pf.data.copy_(sv)
        return output

    def fast_lora_parameters(self):
        """Returns only trainable fast LoRA parameters."""
        return [p for _, p in self._lora_params(self.fast_model)]

    def named_fast_lora_parameters(self):
        return list(self._lora_params(self.fast_model))
