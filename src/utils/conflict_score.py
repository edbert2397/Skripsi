"""
conflict_score.py - Gradient conflict score utilities for O-Conflict (Method 3).

Computes a reference gradient (mean over current-task batches) and scores
candidate samples by negative cosine similarity against that reference:

    score(z) = -cos(g(z), g_ref)

No SVD is needed — only a single mean gradient vector.

Inspired by:
  - A-GEM (Chaudhry et al., ICLR 2019): single reference gradient as proxy
  - CNL (2026): negative gradient similarity as causal mechanism of forgetting
  - GSS (Aljundi et al., NeurIPS 2019): cosine similarity for buffer selection
"""

import torch
from typing import Optional


def estimate_reference_gradient(
    model,
    dataloader,
    device: torch.device,
    n_estimation_batches: int = 10,
    use_fp16: bool = True,
) -> torch.Tensor:
    """
    Phase 1C: compute mean gradient of current task over multiple batches.

    Args:
        model:               DualLoRAModel
        dataloader:          DataLoader for the current task
        device:              CUDA device
        n_estimation_batches: how many batches to average over
        use_fp16:            use autocast fp16 (saves VRAM)

    Returns:
        g_ref of shape [d_lora] on CPU fp32 — the mean gradient vector.
    """
    # Keep model in training mode so gradient checkpointing works correctly.
    # In eval mode, T5 skips the checkpointing guard and enables use_cache=True,
    # which detaches the computation graph and produces zero LoRA gradients.
    was_training = model.training
    model.train()
    m = getattr(model, "fast_model", model)

    for name, param in m.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)

    # Match trainer's dtype: bfloat16 if supported (no overflow risk, no scaler needed),
    # else float16 with GradScaler to prevent gradient underflow.
    amp_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                 else torch.float16)
    use_scaler = use_fp16 and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    sum_grad: Optional[torch.Tensor] = None
    count = 0

    for i, batch in enumerate(dataloader):
        if i >= n_estimation_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        m.zero_grad()

        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = model.compute_loss(batch)
            if scaler is not None:
                scaler.scale(loss).backward()
                inv_scale = 1.0 / scaler.get_scale()
                for name, param in m.named_parameters():
                    if "lora_" in name and param.grad is not None:
                        param.grad.data.mul_(inv_scale)
            else:
                loss.backward()
        else:
            loss = model.compute_loss(batch)
            loss.backward()

        # Extract and flatten LoRA grads to CPU fp32
        grads = []
        for name, param in m.named_parameters():
            if "lora_" in name and param.grad is not None:
                grads.append(param.grad.detach().float().cpu().flatten())

        if grads:
            g = torch.cat(grads)
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            if sum_grad is None:
                sum_grad = g
            else:
                sum_grad = sum_grad + g
            count += 1

        m.zero_grad()

    if sum_grad is None or count == 0:
        raise RuntimeError("No gradients collected for reference gradient estimation.")

    g_ref = sum_grad / count

    if not was_training:
        model.eval()
    torch.cuda.empty_cache()

    return g_ref  # CPU fp32, shape [d_lora]


def gradient_conflict_score(
    grad_vec: torch.Tensor,
    g_ref: torch.Tensor,
) -> float:
    """
    Compute the gradient conflict score for a single sample:

        score(z) = -cos(g(z), g_ref)

    Higher score = more conflict = gradient points opposite to current task.
    These are the samples most at risk of forgetting and should be prioritized
    for replay.

    Args:
        grad_vec:  [d_lora] CPU float32 tensor — per-sample gradient
        g_ref:     [d_lora] CPU float32 tensor — reference (mean) gradient

    Returns:
        Scalar conflict score in [-1, 1].
        +1 means perfectly opposing (highest priority for replay).
        -1 means perfectly aligned (lowest priority).
    """
    norm_g = grad_vec.norm()
    norm_ref = g_ref.norm()

    # Avoid division by zero for degenerate gradients
    if norm_g < 1e-10 or norm_ref < 1e-10:
        return 0.0

    cos_sim = (grad_vec @ g_ref) / (norm_g * norm_ref)
    return -cos_sim.item()
