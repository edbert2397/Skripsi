"""
gradient_utils.py - Per-sample gradient computation for RTX 4050 6GB.

Key techniques to stay within VRAM budget:
  1. Process candidates in small micro-batches (default 8)
  2. Immediately move gradient vectors to CPU after each sample
  3. Use autocast fp16 for forward pass, fp32 for gradient norms
  4. Avoid storing full Jacobians; only extract LoRA grads (tiny)
"""

import torch
from typing import List, Tuple


def extract_lora_grad_vector(model) -> torch.Tensor:
    """
    Flatten all lora_ parameter .grad tensors from the fast model
    into a single CPU float32 vector.

    Returns:
        Tensor of shape [d_lora] on CPU.
    """
    grads = []
    # Work on fast_model if available, else use model directly
    m = getattr(model, "fast_model", model)
    for name, param in m.named_parameters():
        if "lora_" in name and param.grad is not None:
            grads.append(param.grad.detach().float().cpu().flatten())
    if not grads:
        raise RuntimeError(
            "No lora_ gradients found. Did you call loss.backward()?"
        )
    return torch.cat(grads)


@torch.no_grad()
def get_lora_param_vector(model) -> torch.Tensor:
    """Return current fast LoRA weights as flat CPU vector (for reference)."""
    m = getattr(model, "fast_model", model)
    vecs = []
    for name, param in m.named_parameters():
        if "lora_" in name:
            vecs.append(param.detach().float().cpu().flatten())
    return torch.cat(vecs)


def compute_per_sample_gradients(
    model,
    samples: List[dict],
    device: torch.device,
    micro_batch_size: int = 8,
    use_fp16: bool = True,
) -> List[torch.Tensor]:
    """
    Compute the gradient vector (w.r.t. fast LoRA params) for each sample.

    Each sample is processed individually to get a true per-sample gradient.
    We use micro-batching only to amortise CUDA kernel overhead; each
    "micro-batch" here is actually a batch of 1 (real per-sample grads).

    Args:
        model:            DualLoRAModel or OrthogonalLoRAModel
        samples:          list of tokenised dicts {input_ids, attention_mask, labels}
        device:           CUDA device
        micro_batch_size: number of samples to process before clearing cache
        use_fp16:         use autocast for forward (saves VRAM)

    Returns:
        List of CPU float32 tensors, one per sample, each shape [d_lora].
    """
    # Must stay in training mode so gradient checkpointing works and
    # use_cache stays False — eval mode breaks LoRA gradient flow in T5.
    was_training = model.training
    model.train()
    grad_vectors = []

    # Ensure fast LoRA params have grad enabled
    m = getattr(model, "fast_model", model)
    for name, param in m.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)

    for i, sample in enumerate(samples):
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in sample.items()}

        m.zero_grad()

        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model.compute_loss(batch)
        else:
            loss = model.compute_loss(batch)

        loss.backward()

        grad_vec = extract_lora_grad_vector(model)  # CPU fp32
        grad_vectors.append(grad_vec)

        # Periodically clear CUDA cache to avoid fragmentation
        if (i + 1) % micro_batch_size == 0:
            torch.cuda.empty_cache()

    if not was_training:
        model.eval()
    m.zero_grad()
    torch.cuda.empty_cache()
    return grad_vectors
