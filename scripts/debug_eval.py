"""Debug script: compare fast vs slow LoRA generate() output and EMA convergence."""
import torch
import sys
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dual_lora import DualLoRAModel
from src.data.datasets import load_task

device = torch.device("cuda")
base = T5ForConditionalGeneration.from_pretrained("t5-large", local_files_only=True)
tokenizer = T5Tokenizer.from_pretrained("t5-large", legacy=False, local_files_only=True)
model = DualLoRAModel(base, lora_rank=8, lora_alpha=32, beta=0.98).to(device)

train_samples, train_loader, test_loader = load_task(
    "ag_news", tokenizer, 500, 20, batch_size=8
)

# Train for multiple epochs to simulate enough steps
from torch.optim import AdamW

N_EPOCHS = 5
opt = AdamW(model.fast_lora_parameters(), lr=1e-3)
model.train()
n_opt_steps = 0
global_step = 0
for epoch in range(N_EPOCHS):
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model.compute_loss(batch)
        loss.backward()
        if (global_step + 1) % 4 == 0:
            opt.step()
            opt.zero_grad()
            model.ema_update()
            n_opt_steps += 1
        if global_step % 30 == 0:
            print(f"epoch {epoch} step {step} loss={loss.item():.3f}")
        global_step += 1

print(f"\nFinal loss: {loss.item():.3f}, optimizer steps: {n_opt_steps}")

# Check EMA weight distance
fast_params = list(model._lora_params(model.fast_model))
slow_params = list(model._slow_lora)
total_diff = sum(
    (pf.data - ps.data).abs().mean().item()
    for (_, pf), ps in zip(fast_params, slow_params)
) / len(slow_params)
print(f"Avg |fast - slow| per LoRA param: {total_diff:.6f}")
beta_n = 0.98 ** n_opt_steps
print(f"0.98^{n_opt_steps} = {beta_n:.4f} (fraction of initial weights still in slow)")

# --- Eval with generate() ---
model.eval()

# Test with FAST LoRA (swap slow->fast temporarily for generate)
print("\n--- generate() with FAST LoRA (directly trained) ---")
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"][:4].to(device)
        attention_mask = batch["attention_mask"][:4].to(device)
        labels = batch["labels"][:4]
        # Use fast_model.generate directly (no slow swap)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_fast = model.fast_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
            )
        preds_fast = tokenizer.batch_decode(gen_fast, skip_special_tokens=True)
        labels_copy = labels.clone()
        labels_copy[labels_copy == -100] = tokenizer.pad_token_id
        labels_str = tokenizer.batch_decode(labels_copy, skip_special_tokens=True)
        for p, l in zip(preds_fast, labels_str):
            match = "OK" if p.strip().lower() == l.strip().lower() else "MISS"
            print(f"  [{match}] pred={repr(p.strip())} | label={repr(l.strip())}")
        break

# Test with SLOW LoRA (the actual eval path)
print("\n--- generate() with SLOW LoRA (EMA shadow) ---")
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"][:4].to(device)
        attention_mask = batch["attention_mask"][:4].to(device)
        labels = batch["labels"][:4]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_slow = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
            )
        preds_slow = tokenizer.batch_decode(gen_slow, skip_special_tokens=True)
        labels_copy = labels.clone()
        labels_copy[labels_copy == -100] = tokenizer.pad_token_id
        labels_str = tokenizer.batch_decode(labels_copy, skip_special_tokens=True)
        for p, l in zip(preds_slow, labels_str):
            match = "OK" if p.strip().lower() == l.strip().lower() else "MISS"
            print(f"  [{match}] pred={repr(p.strip())} | label={repr(l.strip())}")
        break
