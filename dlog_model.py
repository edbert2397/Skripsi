"""
DLOGModel — wraps a Seq2Seq model by injecting DualLoRA layers into attention projections.

Works with any encoder-decoder model (T5, T5Gemma, FLAN-T5, etc.)
Also provides a baseline wrapper with single LoRA.
"""
import copy
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from typing import List, Dict, Tuple

from dual_lora import DualLoRALinear, SingleLoRALinear


# ======================================================================
# Helper: find and replace linear sub-modules by name pattern
# ======================================================================
def _get_submodules(model: nn.Module, key: str):
    """Walk model tree; return (parent, target, attr_name) for a dotted key."""
    tokens = key.split(".")
    parent = model
    for tok in tokens[:-1]:
        parent = getattr(parent, tok)
    target = getattr(parent, tokens[-1])
    return parent, target, tokens[-1]


def _find_linear_modules(model: nn.Module, target_names: List[str]):
    """
    Find all nn.Linear modules whose leaf name is in *target_names*.
    Returns list of dotted key strings.
    """
    keys = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in target_names:
                keys.append(name)
    return keys


# ======================================================================
# DLOGModel
# ======================================================================
class DLOGModel(nn.Module):
    """
    T5 + Dual-LoRA Orthogonal Gating.

    Injects DualLoRALinear into every attention projection
    specified by *target_modules* (default: ["q", "v"]).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=4,
            torch_dtype="auto", # Use bfloat16/float16 to save 50% VRAM
        )
        if self.base_model.config.pad_token_id is None:
            self.base_model.config.pad_token_id = self.base_model.config.eos_token_id
        
        # Enable gradient checkpointing to save memory (trades compute for memory)
        self.base_model.gradient_checkpointing_enable()

        # Freeze all original parameters, but keep the classification head trainable
        for name, p in self.base_model.named_parameters():
            if "score" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Inject Dual-LoRA layers
        self._dual_lora_keys: List[str] = []
        self._inject_dual_lora(config.target_modules, config.lora_rank, config.lora_alpha)

    # ------------------------------------------------------------------
    def _inject_dual_lora(self, target_names: List[str], rank: int, alpha: float):
        keys = _find_linear_modules(self.base_model, target_names)
        for key in keys:
            parent, old_linear, attr = _get_submodules(self.base_model, key)
            dual = DualLoRALinear(
                in_features=old_linear.in_features,
                out_features=old_linear.out_features,
                rank=rank,
                alpha=alpha,
                bias=old_linear.bias is not None,
            )
            # Copy frozen weights
            dual.weight.data.copy_(old_linear.weight.data)
            if old_linear.bias is not None:
                dual.bias.data.copy_(old_linear.bias.data)
            
            # Match dtype of base model (e.g., bfloat16 for T5Gemma-2)
            dual.to(old_linear.weight.dtype)

            setattr(parent, attr, dual)
            self._dual_lora_keys.append(key)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_dual_lora_layers(self) -> List[DualLoRALinear]:
        """Return all injected DualLoRALinear modules."""
        layers = []
        for key in self._dual_lora_keys:
            _, mod, _ = _get_submodules(self.base_model, key)
            layers.append(mod)
        return layers

    def get_fast_params(self) -> List[nn.Parameter]:
        """Return all Fast LoRA parameters (for optimizer)."""
        params = []
        for layer in self.get_dual_lora_layers():
            params.extend([layer.A_fast, layer.B_fast])
        return params

    def get_slow_params(self) -> List[nn.Parameter]:
        """Return all Slow LoRA parameters."""
        params = []
        for layer in self.get_dual_lora_layers():
            params.extend([layer.A_slow, layer.B_slow])
        return params

    def get_all_lora_params(self) -> List[nn.Parameter]:
        """Return all LoRA params (Fast + Slow) + Classification Head."""
        params = self.get_fast_params() + self.get_slow_params()
        for name, p in self.base_model.named_parameters():
            if "score" in name:
                params.append(p)
        return params

    # ------------------------------------------------------------------
    # EMA consolidation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ema_update_slow(self, decay: float = 0.999):
        for layer in self.get_dual_lora_layers():
            layer.ema_update_slow(decay)

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers for CL phases
    # ------------------------------------------------------------------
    def freeze_slow(self):
        """Freeze Slow LoRA — used during Task-B training."""
        for p in self.get_slow_params():
            p.requires_grad = False

    def unfreeze_slow(self):
        for p in self.get_slow_params():
            p.requires_grad = True

    def unfreeze_all_lora(self):
        for p in self.get_all_lora_params():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Forward (delegates to T5)
    # ------------------------------------------------------------------
    def forward(self, **kwargs):
        return self.base_model(**kwargs)


# ======================================================================
# BaselineModel — single LoRA (no gating)
# ======================================================================
class BaselineModel(nn.Module):
    """
    T5 + Single-LoRA baseline for comparison.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=4,
            torch_dtype="auto",
        )
        if self.base_model.config.pad_token_id is None:
            self.base_model.config.pad_token_id = self.base_model.config.eos_token_id
        
        # Enable gradient checkpointing to save memory
        self.base_model.gradient_checkpointing_enable()

        # Freeze all original parameters, but keep the classification head trainable
        for name, p in self.base_model.named_parameters():
            if "score" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self._lora_keys: List[str] = []
        self._inject_single_lora(config.target_modules, config.lora_rank, config.lora_alpha)

    def _inject_single_lora(self, target_names: List[str], rank: int, alpha: float):
        keys = _find_linear_modules(self.base_model, target_names)
        for key in keys:
            parent, old_linear, attr = _get_submodules(self.base_model, key)
            single = SingleLoRALinear(
                in_features=old_linear.in_features,
                out_features=old_linear.out_features,
                rank=rank,
                alpha=alpha,
                bias=old_linear.bias is not None,
            )
            single.weight.data.copy_(old_linear.weight.data)
            if old_linear.bias is not None:
                single.bias.data.copy_(old_linear.bias.data)
            
            # Match dtype of base model
            single.to(old_linear.weight.dtype)

            setattr(parent, attr, single)
            self._lora_keys.append(key)

    def get_lora_params(self) -> List[nn.Parameter]:
        params = []
        for key in self._lora_keys:
            _, mod, _ = _get_submodules(self.base_model, key)
            params.extend([mod.A, mod.B])
        
        # Add classification head params
        for name, p in self.base_model.named_parameters():
            if "score" in name:
                params.append(p)
                
        return params

    def forward(self, **kwargs):
        return self.base_model(**kwargs)
