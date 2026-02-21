# DLOG Vibe Coding Experiment 🚀

## The "Crash & Rescue" Demo

This experiment demonstrates catastrophic forgetting and how DLOG rescues it using **orthogonal gating**.

### The Setup

**Task A: SST-2 (Sentiment Analysis)** 🎭  
- Input: `"sentiment: The movie was wonderful."`
- Output: `"Positive"`
- What the model learns: Words like "wonderful" → Positive, "dull" → Negative

**Task B: AG News (News Classification)** 📰  
- Input: `"classify: Apple released new phone"`
- Output: `"Sci/Tech"`
- What the model learns: Words like "Apple", "phone" → Sci/Tech, "soccer" → Sports

---

## The Crisis: What Goes Wrong?

### Without DLOG (Standard Sequential Finetuning)
```python
# Phase 1: Train on Sentiment
Model learns: "wonderful" → Positive ✅

# Phase 2: Train on News
Model learns: "Apple" → Sci/Tech ✅
BUT: Gradients for "Apple" accidentally overwrite neurons storing "wonderful"
Result: Model outputs garbage for "wonderful" 💥 CATASTROPHIC FORGETTING

# Test Phase:
Input: "The movie was wonderful."
Expected: "Positive"
Actual: "World" or "Sci/Tech" ❌ (completely wrong!)
```

### With DLOG (Orthogonal Gating)
```python
# Phase 1: Train on Sentiment
Slow LoRA learns: "wonderful" → Positive ✅
Fast LoRA: mirrors Slow

# Phase 2: Train on News (THE MAGIC HAPPENS HERE)
1. Fast LoRA wants to learn: "Apple" → Sci/Tech
2. Compute gradient: ∇θ_fast
3. 🛡️ ORTHOGONAL PROJECTION:
   - Retrieve Slow LoRA's memory subspace (sentiment knowledge)
   - Project: ∇θ_fast⊥ = ∇θ_fast - Proj_Slow(∇θ_fast)
   - Translation: "Hey Fast LoRA, learn whatever you want, 
                  EXCEPT don't touch the directions Slow uses for sentiment!"
4. Apply clean gradient: ∇θ_fast⊥
5. EMA consolidation: Slow ← 0.999·Slow + 0.001·Fast

# Test Phase:
Input: "The movie was wonderful."
Expected: "Positive"
Actual: "Positive" ✅ (protected by orthogonality!)

Input: "Apple released new phone"
Expected: "Sci/Tech"
Actual: "Sci/Tech" ✅ (new knowledge learned!)
```

---

## Quick Start (Vibe Coding)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Smoke Test (2 minutes, proves it works)
```bash
cd e:\Kuliah\sem8\skripsi\dlog
python run_experiment.py --smoke-test
```

This runs **5 training steps per task** with tiny data to verify:
- ✅ Data loads correctly (SST-2 → AG News)
- ✅ Orthogonal projection works (leakage → 0)
- ✅ Both DLOG and Baseline complete without errors

### 3️⃣ Full Experiment (~30-45 min on RTX GPU)
```bash
python run_experiment.py --steps 1000
```

Expected results:
- **DLOG**: FP ≈ 0.75-0.85, FT ≈ 0.05-0.15 (low forgetting)
- **Baseline**: FP ≈ 0.60-0.70, FT ≈ 0.20-0.35 (high forgetting)

### 4️⃣ Ablation Study (test Soft vs Hard constraints)
```bash
python run_experiment.py --ablation --steps 1000
```

Compares 5 variants:
1. Soft constraint only
2. Hard constraint only (parameter-subspace)
3. Hard constraint only (memory-gradient) ⭐ recommended
4. Soft + Hard (parameter)
5. Soft + Hard (memory-gradient) ⭐ recommended

---

## Output Files

All saved to `dlog/results/<model_name>/`:

| File | What It Shows |
|------|---------------|
| `forgetting_comparison.png` | Bar chart: DLOG vs Baseline (FP & FT) |
| `training_loss.png` | Loss curves over training |
| `leakage.png` | Leakage ratio (should drop to ~0 for DLOG) |
| `performance_matrix.png` | Heatmap R[i][j] = accuracy on task j after training task i |
| `efficiency_table.png` | Wall-clock time, forward passes, memory usage |
| `results_table.txt` | Summary metrics (copy-paste to thesis!) |

---

## What to Look For (Vibe Check ✅)

1. **Leakage Plot**: Should drop to ~0.00 after orthogonal projection kicks in
2. **Performance Matrix**: 
   - DLOG: R[1][0] ≈ R[0][0] (no forgetting on Task A)
   - Baseline: R[1][0] << R[0][0] (severe forgetting on Task A)
3. **Forgetting Metric**: DLOG << Baseline

---

## Config Tweaks

Edit `config.py` to adjust:
```python
lora_rank = 8              # LoRA rank (try 4, 8, 16)
lambda_orth = 0.1          # Soft constraint weight
projection_type = "memory_gradient"  # or "parameter"
ema_decay = 0.999          # Slow LoRA consolidation rate
replay_ratio = 0.2         # 20% replay in each batch
```

---

## GPU Requirements

- **Minimum**: RTX 3060 (12GB) or equivalent
- **T5-Large**: ~770M params, needs ~8-10GB VRAM with FP16
- **Smoke test**: Runs on CPU (slow, for testing only)

If you get OOM errors:
```bash
# Use T5-Base instead (220M params, ~3-4GB VRAM)
python run_experiment.py --model t5-base
```

---

## The Math (Quick Reference)

**Orthogonal Projection** (the core trick):
```
Given:
  - g_fast = gradient of Fast LoRA
  - U = orthonormal basis from Slow LoRA (or memory gradients)

Project:
  g_fast⊥ = g_fast - U @ U^T @ g_fast

Proof of orthogonality:
  <g_fast⊥, U> = <g_fast - U @ U^T @ g_fast, U>
                = <g_fast, U> - <U @ U^T @ g_fast, U>
                = <g_fast, U> - <g_fast, U @ U^T @ U>
                = <g_fast, U> - <g_fast, U>  (since U^T @ U = I)
                = 0  ✅
```

**EMA Consolidation**:
```
θ_slow ← α·θ_slow + (1-α)·θ_fast  (α = 0.999)
```

This slowly "distills" Fast's new knowledge into Slow's long-term memory.

---

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'transformers'`  
**Fix**: `pip install -r requirements.txt`

**Problem**: CUDA out of memory  
**Fix**: Use smaller batch size or T5-base: `python run_experiment.py --model t5-base`

**Problem**: Training stuck at 0% for minutes  
**Fix**: First run downloads datasets (~500MB). Wait or check internet connection.

**Problem**: Leakage not dropping to 0  
**Fix**: This is expected for `--projection parameter` mode. Use `--projection memory_gradient` (default).

---

Happy vibe coding! 🎉
