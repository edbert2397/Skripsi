"""
feature_subspace.py - Feature subspace estimation via truncated SVD.

O-Feat counterpart to subspace.py (O-Grad).  The subspace is built from
encoder hidden states instead of gradients:

  Phase 1B:  H = [all non-padding token features]  ∈ R^{n_tokens × d_hidden}
             Truncated SVD → V_k ∈ R^{d_hidden × k}

  Phase 2B:  score(z) = ||h(z) - V_k V_k^T h(z)||_2
             where h(z) = mean-pooled encoder representation of sample z.

Memory profile (T5-Large, d_hidden=1024):
  H ≈ [10_000, 1024] ≈ 40 MB — easily fits in CPU RAM.
  V_k ≈ [1024, 10] ≈ 40 KB.
"""

import torch
from typing import Optional, Tuple


def estimate_feature_subspace(
    model,
    dataloader,
    device: torch.device,
    subspace_rank_k: int = 10,
    n_estimation_batches: int = 10,
    use_fp16: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phase 1B: estimate the current task's feature subspace from encoder
    hidden states.

    Collects non-padding token representations from n_estimation_batches,
    then computes the top-k right singular vectors via SVD.

    Args:
        model:               DualLoRAModel (or OrthogonalLoRAModel)
        dataloader:          DataLoader for the current task
        device:              CUDA device
        subspace_rank_k:     k for truncated SVD
        n_estimation_batches: how many batches to use
        use_fp16:            use autocast fp16

    Returns:
        V_k of shape [d_hidden, k] on CPU fp32 — top-k right singular vectors.
    """
    model.eval()
    m = getattr(model, "fast_model", model)

    all_features = []

    for i, batch in enumerate(dataloader):
        if i >= n_estimation_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.no_grad():
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = m(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        output_hidden_states=True,
                    )
            else:
                outputs = m(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=True,
                )

        # encoder_last_hidden_state: [B, seq_len, d_hidden]
        hidden = outputs.encoder_last_hidden_state.float().cpu()
        mask = batch["attention_mask"].cpu()  # [B, seq_len]

        # Collect only non-padding token features
        for b in range(hidden.size(0)):
            valid_len = int(mask[b].sum().item())
            if valid_len > 0:
                all_features.append(hidden[b, :valid_len, :])  # [valid_len, d_hidden]

    if not all_features:
        raise RuntimeError("No features collected for subspace estimation.")

    # H: [total_tokens, d_hidden]  (all on CPU)
    H = torch.cat(all_features, dim=0)
    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    k = min(subspace_rank_k, min(H.shape))

    if H.shape[0] <= H.shape[1]:
        # Few samples relative to d_hidden — use Gram matrix trick
        # (same approach as O-Grad's subspace.py)
        HHT = H @ H.T  # [n_tokens, n_tokens]
        HHT = torch.nan_to_num(HHT, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            eigenvalues, U = torch.linalg.eigh(HHT)
        except torch.linalg.LinAlgError:
            HHT = HHT + 1e-8 * torch.eye(HHT.shape[0])
            eigenvalues, U = torch.linalg.eigh(HHT)

        U_k = U[:, -k:]                                      # [n_tokens, k]
        sigma_k = eigenvalues[-k:].clamp(min=1e-10).sqrt()    # [k]
        V_k = (H.T @ U_k) / sigma_k.unsqueeze(0)             # [d_hidden, k]
        all_sigmas = eigenvalues.clamp(min=0).sqrt().flip(0)  # full spectrum, descending
    else:
        # Many tokens (typical case) — use randomized SVD (efficient for top-k)
        U, S, V = torch.svd_lowrank(H, q=k, niter=4)
        V_k = V  # [d_hidden, k]
        all_sigmas = S   # svd_lowrank already returns top-k in descending order

    V_k = V_k.contiguous()

    model.train()
    torch.cuda.empty_cache()

    return V_k, all_sigmas


def feature_orthogonal_residual_norm(
    feature_vec: torch.Tensor,
    V_k: torch.Tensor,
) -> float:
    """
    Compute ||h_perp|| = ||h - V_k V_k^T h|| for a single feature vector.

    Identical to subspace.orthogonal_residual_norm but operates in feature
    space (d_hidden ≈ 1024) rather than gradient space (d_lora ≈ 100K).

    Args:
        feature_vec: [d_hidden] CPU float32 tensor (mean-pooled encoder output)
        V_k:         [d_hidden, k] CPU float32 tensor

    Returns:
        Scalar orthogonal residual norm.
    """
    coords = V_k.T @ feature_vec       # [k]
    h_parallel = V_k @ coords          # [d_hidden]
    h_perp = feature_vec - h_parallel
    return h_perp.norm().item()
