"""
subspace.py - Gradient subspace estimation via truncated SVD.

Adapted for RTX 4050 6GB: gradient matrix G is built on CPU (it's
[n_batches x d_lora] ≈ [10 x ~100K] for rank-8 LoRA, easily fits in RAM),
then SVD is performed on CPU with torch.linalg.svd.
"""

import torch
from typing import Optional


def estimate_gradient_subspace(
    model,
    dataloader,
    device: torch.device,
    subspace_rank_k: int = 10,
    n_estimation_batches: int = 10,
    use_fp16: bool = True,
) -> torch.Tensor:
    """
    Phase 1 of the method: estimate the current task's gradient subspace.

    Computes batch-level gradients (not per-sample) for efficiency.
    Each row of G is the flattened gradient for one mini-batch.

    Args:
        model:               DualLoRAModel
        dataloader:          DataLoader for the current task
        device:              CUDA device
        subspace_rank_k:     k for truncated SVD (number of basis vectors)
        n_estimation_batches: how many batches to use
        use_fp16:            use autocast fp16 (saves VRAM)

    Returns:
        V_k of shape [d_lora, k] on CPU — the top-k right singular vectors.
        (Projection is applied lazily as V_k @ (V_k.T @ g) to avoid [d_lora, d_lora] matrix.)
    """
    model.eval()
    m = getattr(model, "fast_model", model)

    for name, param in m.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)

    all_grads = []

    for i, batch in enumerate(dataloader):
        if i >= n_estimation_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        m.zero_grad()

        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model.compute_loss(batch)
        else:
            loss = model.compute_loss(batch)

        loss.backward()

        # Extract and flatten LoRA grads → CPU fp32
        grads = []
        for name2, param in m.named_parameters():
            if "lora_" in name2 and param.grad is not None:
                grads.append(param.grad.detach().float().cpu().flatten())
        if grads:
            g = torch.cat(grads)
            # Sanitize: replace inf/nan (fp16 overflow) with 0 before SVD
            if not torch.isfinite(g).all():
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            all_grads.append(g)

        m.zero_grad()

    if not all_grads:
        raise RuntimeError("No gradients collected for subspace estimation.")

    # G: [n_batches, d_lora]  (all on CPU)
    G = torch.stack(all_grads)
    G = torch.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)

    # Gram matrix trick: instead of SVD on [n_batches, d_lora] (huge LAPACK workspace
    # → segfault), eigendecompose the tiny [n_batches, n_batches] gram matrix.
    #   G @ G.T = U Λ U.T  (eigendecomposition, cheap)
    #   Right singular vectors: V_k = G.T @ U_k / sqrt(λ_k)
    GGT = G @ G.T  # [n_batches, n_batches]
    GGT = torch.nan_to_num(GGT, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        eigenvalues, U = torch.linalg.eigh(GGT)  # ascending order, [n_batches]
    except torch.linalg.LinAlgError:
        GGT = GGT + 1e-8 * torch.eye(GGT.shape[0])
        eigenvalues, U = torch.linalg.eigh(GGT)

    k = min(subspace_rank_k, G.shape[0])
    # eigh returns ascending → take last k columns for top-k
    U_k = U[:, -k:]                              # [n_batches, k]
    sigma_k = eigenvalues[-k:].clamp(min=1e-10).sqrt()  # [k]
    V_k = (G.T @ U_k) / sigma_k.unsqueeze(0)    # [d_lora, k]
    V_k = V_k.contiguous()

    model.train()
    torch.cuda.empty_cache()

    return V_k   # CPU fp32, shape [d_lora, k]


def orthogonal_residual_norm(
    grad_vec: torch.Tensor,
    V_k: torch.Tensor,
) -> float:
    """
    Compute ||g_perp|| = ||g - V_k V_k^T g|| for a single gradient vector.

    Uses the factored form V_k @ (V_k.T @ g) to avoid materializing [d_lora, d_lora].

    Args:
        grad_vec:  [d_lora] CPU float32 tensor
        V_k:       [d_lora, k] CPU float32 tensor (top-k right singular vectors)

    Returns:
        Scalar orthogonal residual norm.
    """
    coords = V_k.T @ grad_vec          # [k]
    g_parallel = V_k @ coords          # [d_lora]
    g_perp = grad_vec - g_parallel
    return g_perp.norm().item()
