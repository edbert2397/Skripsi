"""
DualLoRALinear — the core building block of DLOG.

Replaces a single nn.Linear with:
    W_total = W_frozen + B_s @ A_s  (Slow/Memory)  +  B_f @ A_f  (Fast/Task)

Slow LoRA consolidates learned knowledge; Fast LoRA adapts to new tasks.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DualLoRALinear(nn.Module):
    """
    A linear layer augmented with two LoRA branches (Slow + Fast).

    Parameters
    ----------
    in_features : int
        Input dimension of the original linear layer.
    out_features : int
        Output dimension of the original linear layer.
    rank : int
        Low-rank dimension for both LoRA branches.
    alpha : float
        LoRA scaling factor  (scaling = alpha / rank).
    bias : bool
        Whether the original layer has bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # --- Frozen base weight (loaded later) ---
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        # --- Slow LoRA (Memory) ---
        self.A_slow = nn.Parameter(torch.empty(rank, in_features))
        self.B_slow = nn.Parameter(torch.empty(out_features, rank))

        # --- Fast LoRA (Task) ---
        self.A_fast = nn.Parameter(torch.empty(rank, in_features))
        self.B_fast = nn.Parameter(torch.empty(out_features, rank))

        self.reset_lora_parameters()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def reset_lora_parameters(self):
        """
        Slow LoRA: Kaiming init (active from start).
        Fast LoRA: Zero-initialized so output contribution = 0 at t=0,
                   preserving the model's original behavior (residual gating).
        """
        # Slow
        nn.init.kaiming_uniform_(self.A_slow, a=math.sqrt(5))
        nn.init.zeros_(self.B_slow)

        # Fast — zero init (no contribution at start)
        nn.init.zeros_(self.A_fast)
        nn.init.zeros_(self.B_fast)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        y = W_frozen @ x  +  scaling * (B_s @ A_s @ x)  +  scaling * (B_f @ A_f @ x)
        """
        # Base
        out = F.linear(x, self.weight, self.bias)

        # Slow LoRA
        slow = F.linear(F.linear(x, self.A_slow), self.B_slow) * self.scaling

        # Fast LoRA
        fast = F.linear(F.linear(x, self.A_fast), self.B_fast) * self.scaling

        return out + slow + fast

    # ------------------------------------------------------------------
    # Helpers for gating
    # ------------------------------------------------------------------
    def get_slow_params(self):
        """Return (A_slow, B_slow) for orthogonal gating computations."""
        return self.A_slow, self.B_slow

    def get_fast_params(self):
        """Return (A_fast, B_fast) for orthogonal gating computations."""
        return self.A_fast, self.B_fast

    # ------------------------------------------------------------------
    # EMA update: θ_slow ← decay * θ_slow + (1 - decay) * θ_fast
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ema_update_slow(self, decay: float = 0.999):
        """Consolidate Fast → Slow via exponential moving average."""
        self.A_slow.data.mul_(decay).add_(self.A_fast.data, alpha=1.0 - decay)
        self.B_slow.data.mul_(decay).add_(self.B_fast.data, alpha=1.0 - decay)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


class SingleLoRALinear(nn.Module):
    """
    Baseline: a single LoRA branch (no Slow/Fast separation).
    Used for the SeqFT + Single LoRA baseline comparison.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.A), self.B) * self.scaling
        return out + lora

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}"
        )
