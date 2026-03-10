# Orthogonal Gradient Replay for Class-Incremental Learning

Implementation of *Orthogonal Gradient-Based Replay Selection with Slow-Fast LoRA for Class-Incremental Learning* — undergraduate thesis by Edbert, based on the SuRe paper (Hazard et al., 2025).

---

## Hardware: RTX 4050 6GB

All defaults are tuned for 6GB VRAM:

| Setting | Value | Why |
|---|---|---|
| `batch_size_current` | 8 | Reduced from 64 |
| `batch_size_replay` | 4 | Reduced from 32 |
| `grad_accum_steps` | 8 | Effective batch = 64 |
| `fp16` | True | Mixed precision saves ~40% VRAM |
| `gradient_checkpointing` | True | Saves ~40% activation memory |
| `grad_batch_size` | 8 | Per-sample grad micro-batch |

T5-Large + LoRA rank 8 + fp16 + gradient checkpointing ≈ **4.5–5.5 GB** peak.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Default run (Slow Orthogonal, standard_cl, order 0, seed 42):
```bash
python scripts/run_experiment.py
```

### SuRe baselines:
```bash
# Reservoir replay (no selection, no EMA)
python scripts/run_experiment.py --selection reservoir --no-ema

# Surprise replay (SuRe selection only)
python scripts/run_experiment.py --selection surprise --no-ema

# Slow Surprise (SuRe full method)
python scripts/run_experiment.py --selection surprise
```

### Our methods:
```bash
# Orthogonal Replay (our selection, no EMA)
python scripts/run_experiment.py --selection orthogonal --no-ema

# Slow Orthogonal (core contribution: orthogonal + EMA)
python scripts/run_experiment.py --selection orthogonal

# Triple Defense (Insight 6: orthogonal + O-LoRA + EMA)
python scripts/run_experiment.py --selection orthogonal --use-orthogonal-lora
```

### LNT benchmark, 3 orders:
```bash
python scripts/run_experiment.py --benchmark lnt --order 0
python scripts/run_experiment.py --benchmark lnt --order 1
python scripts/run_experiment.py --benchmark lnt --order 2
```

### Orthogonal-Before variant (buffer update before training):
```bash
python scripts/run_experiment.py --update-buffer-before
```

---

## Ablation Studies

Each ablation corresponds to a section in the thesis (§7).

### Run a single ablation:
```bash
python scripts/run_ablation.py --ablation-subspace-rank    # §7.1 k ∈ {5,10,20,50}
python scripts/run_ablation.py --ablation-buffer-size      # §7.2 buffer ∈ {150,300,500,1500}
python scripts/run_ablation.py --ablation-replay-ratio     # §7.3 ratio ∈ {1:2,1:4,1:8,1:16}
python scripts/run_ablation.py --ablation-ema-beta         # §7.4 β ∈ {0.985,0.99,0.995,0.999}
python scripts/run_ablation.py --ablation-timing           # §7.5 Before vs After
python scripts/run_ablation.py --ablation-hybrid           # §7.6 α ∈ {0,0.25,0.5,0.75,1.0}
```

### Run ALL ablations sequentially:
```bash
python scripts/run_ablation.py --ablation-all
```

### Quick sanity check (1 seed, 1 order, standard_cl):
```bash
python scripts/run_ablation.py --ablation-subspace-rank --fast
```

### Custom seeds/orders:
```bash
python scripts/run_ablation.py --ablation-buffer-size --seeds 42 123 --orders 0 1
```

### Without any flag (shows usage):
```bash
python scripts/run_ablation.py   # prints usage and exits cleanly
```

---

## Plotting Results

```bash
# Full main results table + all ablation plots
python scripts/plot_results.py --results-dir outputs/

# Single ablation plot
python scripts/plot_results.py --results-dir outputs/ --ablation subspace_rank
```

---

## Project Structure

```
orthogonal-replay/
├── configs/
│   ├── base.yaml              # Shared hyperparameters (RTX 4050 defaults)
│   ├── standard_cl.yaml       # 4-task benchmark
│   └── lnt.yaml               # 15-task benchmark
├── src/
│   ├── models/
│   │   ├── dual_lora.py       # Fast/slow LoRA + EMA
│   │   └── orthogonal_lora.py # O-LoRA subspace allocation (Insight 6)
│   ├── selection/
│   │   ├── base.py            # Abstract selector
│   │   ├── orthogonal.py      # ← Core contribution: ||g_⊥|| scoring
│   │   ├── surprise.py        # SuRe NLL selection
│   │   ├── reservoir.py       # Random baseline
│   │   └── hybrid.py          # α·||g_⊥|| + (1-α)·NLL
│   ├── buffer/
│   │   └── replay_buffer.py   # Fixed-size buffer, per-task quotas
│   ├── training/
│   │   ├── trainer.py         # Main CL loop
│   │   ├── evaluator.py       # FP / AP / Forgetting metrics
│   │   └── ema.py             # EMA step
│   ├── data/
│   │   ├── datasets.py        # All 15 task loaders
│   │   └── task_orders.py     # Reproducible task orderings
│   └── utils/
│       ├── gradient_utils.py  # Per-sample gradient computation
│       ├── subspace.py        # SVD subspace estimation
│       └── logging.py         # JSON + wandb logging
└── scripts/
    ├── run_experiment.py      # Main entry point
    ├── run_ablation.py        # --ablation-* flags
    └── plot_results.py        # Tables + figures
```

---

## Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--benchmark` | `standard_cl` | `standard_cl` or `lnt` |
| `--order` | `0` | Task order (0, 1, 2) |
| `--seed` | `42` | Random seed |
| `--selection` | `orthogonal` | `orthogonal`, `surprise`, `reservoir`, `hybrid` |
| `--no-ema` | — | Disable EMA consolidation |
| `--use-orthogonal-lora` | — | Enable O-LoRA (Triple Defense) |
| `--beta` | `0.995` | EMA decay rate |
| `--buffer-size` | `300` | Total replay buffer size |
| `--replay-ratio` | `1:2` | Replay to current sample ratio |
| `--subspace-rank-k` | `10` | SVD rank for subspace estimation |
| `--hybrid-alpha` | `0.5` | Weight for orthogonal score in hybrid |
| `--update-buffer-before` | — | Use Orthogonal-Before variant |
| `--fast` | — | (ablation only) Quick 1-seed check |
