"""
Orthogonal Gating — the core DLOG innovation.

Two mechanisms:
  1. Soft Constraint:  L_orth = Σ ||A_f A_s^T||²_F + ||B_f^T B_s||²_F
  2. Hard Constraint:  Gradient projection g_f⊥ = g_f - U U^T g_f
     - Option A: U from Slow LoRA parameters  (parameter-subspace)
     - Option B: U from memory-gradient basis   (memory-gradient subspace)
"""
import torch
import torch.nn as nn
from typing import List, Optional
from collections import deque

from dual_lora import DualLoRALinear


# ======================================================================
# 1. Soft Orthogonality Constraint (Loss Regularization)
# ======================================================================
def compute_orth_loss(dual_layers: List[DualLoRALinear]) -> torch.Tensor:
    """
    Compute orthogonality penalty across all Dual-LoRA layers.

    L_orth = Σ_i (||A_f_i @ A_s_i^T||²_F  +  ||B_f_i^T @ B_s_i||²_F)
    """
    loss = torch.tensor(0.0, device=dual_layers[0].A_slow.device)
    for layer in dual_layers:
        A_s, B_s = layer.get_slow_params()
        A_f, B_f = layer.get_fast_params()

        # A interaction: [r x d] @ [d x r] = [r x r]
        loss = loss + torch.norm(A_f @ A_s.T, p="fro") ** 2
        # B interaction: [r x d_out]^T @ [d_out x r] => [r x r]
        loss = loss + torch.norm(B_f.T @ B_s, p="fro") ** 2

    return loss


# ======================================================================
# 2. Hard Orthogonality — Parameter-Subspace Projection
# ======================================================================
@torch.no_grad()
def project_gradients_parameter(dual_layers: List[DualLoRALinear]):
    """
    Simplified Sherman-Morrison projection (from proposal Section 3.1).
    
    P_slow = I - (A_s @ A_s^T) / (||A_s||^2 + λ)
    ∇θ_fast⊥ = P_slow @ ∇θ_fast  (projection in rank space)
    
    This is computationally cheaper than full Gram matrix inversion
    and matches the thesis mathematical formulation exactly.
    """
    lambda_reg = 1e-6
    
    for layer in dual_layers:
        A_s, B_s = layer.get_slow_params()
        A_f, B_f = layer.get_fast_params()

        # --- Project A_fast gradient ---
        if A_f.grad is not None:
            g = A_f.grad  # [r, d]
            # Sherman-Morrison simplified projection
            # P_slow = I - (A_s @ A_s^T) / (||A_s||^2 + λ)
            norm_sq = torch.norm(A_s) ** 2
            P_slow = torch.eye(A_s.shape[0], device=A_s.device) - \
                     (A_s @ A_s.T) / (norm_sq + lambda_reg)
            A_f.grad = P_slow @ g  # Apply projection in rank space

        # --- Project B_fast gradient ---
        if B_f.grad is not None:
            g = B_f.grad  # [d_out, r]
            # Same formula for B matrices (project on right side)
            norm_sq = torch.norm(B_s) ** 2
            P_slow = torch.eye(B_s.shape[1], device=B_s.device) - \
                     (B_s.T @ B_s) / (norm_sq + lambda_reg)
            B_f.grad = g @ P_slow  # Apply projection in rank space


# ======================================================================
# 3. Hard Orthogonality — Memory-Gradient Subspace Projection
# ======================================================================
class MemoryGradientProjector:
    """
    Option 2 (Recommended): Build a protected subspace from memory (replay)
    gradients instead of raw Slow parameters.

    Maintains a low-rank orthonormal basis U_m via incremental QR over
    the last K steps of replay-batch gradients.

    Usage:
        projector = MemoryGradientProjector(buffer_size=10)

        # During training:
        projector.accumulate(fast_params)       # after backward on replay batch
        projector.project_gradients(fast_params) # after backward on new-task batch
    """

    def __init__(self, buffer_size: int = 10, max_basis_rank: int = 50):
        self.buffer_size = buffer_size
        self.max_basis_rank = max_basis_rank
        # Separate buffer per parameter (keyed by id)
        self._grad_buffers: dict = {}   # param_id -> deque of flat grad vectors
        self._bases: dict = {}          # param_id -> orthonormal basis U [d, k]

    @torch.no_grad()
    def accumulate(self, params: List[nn.Parameter]):
        """
        Store current .grad for each param into the rolling buffer,
        then recompute the orthonormal basis via QR.
        """
        for p in params:
            pid = id(p)
            if p.grad is None:
                continue
            g = p.grad.detach().clone().flatten()

            if pid not in self._grad_buffers:
                self._grad_buffers[pid] = deque(maxlen=self.buffer_size)
            self._grad_buffers[pid].append(g)

            # Recompute basis via QR of stacked gradients
            G = torch.stack(list(self._grad_buffers[pid]), dim=1)  # [d, K]
            # Truncate to max_basis_rank columns
            if G.shape[1] > self.max_basis_rank:
                G = G[:, -self.max_basis_rank:]
            
            # QR decomposition doesn't support BFloat16, convert temporarily
            orig_dtype = G.dtype
            if orig_dtype == torch.bfloat16:
                G = G.float()
            
            Q, R = torch.linalg.qr(G, mode="reduced")  # Q: [d, k]
            
            # Convert back to original dtype
            if orig_dtype == torch.bfloat16:
                Q = Q.bfloat16()
            
            self._bases[pid] = Q

    @torch.no_grad()
    def project_gradients(self, params: List[nn.Parameter]):
        """
        Project each param's .grad to remove components in the
        memory-gradient subspace:  g⊥ = g - U_m @ U_m^T @ g
        """
        for p in params:
            pid = id(p)
            if p.grad is None or pid not in self._bases:
                continue
            g_flat = p.grad.flatten()
            U = self._bases[pid]  # [d, k]

            # Project out memory components
            coeffs = U.T @ g_flat       # [k]
            proj = U @ coeffs            # [d]
            g_projected = g_flat - proj

            p.grad = g_projected.view_as(p.grad)

    def compute_leakage(self, params: List[nn.Parameter]) -> float:
        """
        Compute leakage metric:
            Leak = ||U U^T g_f|| / ||g_f||
        Returns average leakage across all params.
        """
        leakages = []
        for p in params:
            pid = id(p)
            if p.grad is None or pid not in self._bases:
                continue
            g_flat = p.grad.detach().flatten()
            U = self._bases[pid]
            proj_norm = torch.norm(U @ (U.T @ g_flat))
            g_norm = torch.norm(g_flat)
            if g_norm > 1e-8:
                leakages.append((proj_norm / g_norm).item())
        return sum(leakages) / max(len(leakages), 1)

    def reset(self):
        """Clear all accumulated gradients and bases."""
        self._grad_buffers.clear()
        self._bases.clear()


# ======================================================================
# Convenience dispatcher
# ======================================================================
def apply_hard_projection(
    dual_layers: List[DualLoRALinear],
    projector: Optional[MemoryGradientProjector] = None,
    projection_type: str = "parameter",
):
    """
    Apply gradient projection based on the chosen strategy.

    projection_type:
      - "parameter"      : project against Slow LoRA params (Option 1)
      - "memory_gradient" : project against memory-gradient basis (Option 2)
    """
    if projection_type == "parameter":
        project_gradients_parameter(dual_layers)
    elif projection_type == "memory_gradient":
        if projector is None:
            raise ValueError("MemoryGradientProjector required for memory_gradient mode")
        fast_params = []
        for layer in dual_layers:
            fast_params.extend([layer.A_fast, layer.B_fast])
        projector.project_gradients(fast_params)
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")
