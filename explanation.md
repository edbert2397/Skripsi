# DLOG System Explanation
### How Everything Works — Forward, Backward, Replay, Orthogonalization, and Freezing

---

## 1. Big Picture

DLOG trains a language model on a sequence of tasks **one at a time**, without revisiting old task data (except for a small replay buffer). The challenge is **Catastrophic Forgetting**: when you train on Task 2, the model tends to overwrite what it learned from Task 1.

DLOG prevents this using three mechanisms working together:

```
┌──────────────────────────────────────────────────────────┐
│                    DLOG Training Loop                    │
│                                                          │
│  For each task T:                                        │
│    ┌─────────────┐    ┌──────────────┐    ┌──────────┐  │
│    │   Backward  │    │    Replay    │    │   IMC    │  │
│    │  (Orth.     │ +  │  (Random     │ +  │ Freezing │  │
│    │  Projection)│    │   Samples)   │    │          │  │
│    └─────────────┘    └──────────────┘    └──────────┘  │
│                              │                           │
│                    Task Boundary: Consolidate            │
│                    (Fast → Slow, then reset Fast)        │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Architecture: How Slow-Fast Integrates with the Backbone

### 2.1 The Base Model

DLOG uses **Qwen2.5-1.5B** loaded as a Sequence Classifier (`AutoModelForSequenceClassification`). All original weights are **completely frozen** — the base model never changes during training.

The model has 28 Transformer layers. Each layer contains attention projections:
- `q_proj` (Query): maps hidden states → query vectors
- `v_proj` (Value): maps hidden states → value vectors

These are the only layers DLOG injects into (LoRA targets).

### 2.2 Injecting Dual-LoRA into Every Target Layer

For each target projection (e.g., `q_proj` in layer 3), DLOG **replaces** the original `nn.Linear` with a `DualLoRALinear`:

```
Before injection:
    q_proj = nn.Linear(1536, 1536)   ← frozen original

After injection:
    q_proj = DualLoRALinear(
        weight  = [original frozen weights]     ← never changes
        A_slow  = [r × d_in]  = [8 × 1536]    ← memory branch
        B_slow  = [d_out × r] = [1536 × 8]    ← memory branch
        A_fast  = [r × d_in]  = [8 × 1536]    ← task branch
        B_fast  = [d_out × r] = [1536 × 8]    ← task branch
    )
```

With Qwen2.5-1.5B having 28 layers and 2 targets (q_proj, v_proj), DLOG injects:
```
28 layers × 2 targets = 56 DualLoRALinear modules
```

Each module has 4 LoRA matrices: A_slow, B_slow, A_fast, B_fast.

### 2.3 Initialization

| Parameter | Init | Reason |
|-----------|------|--------|
| `A_slow` | **Zero** | Slow must be completely silent in Task 1 |
| `B_slow` | **Zero** | Same — gradient chain is algebraically broken |
| `A_fast` | **Kaiming uniform** | Standard LoRA init — receives gradients immediately |
| `B_fast` | **Zero** | Standard LoRA init — output starts at 0 but A gets gradients |

**Why zero-init for Slow?** If Slow started with non-zero values, it would train alongside Fast in Task 1, making them nearly identical. Then when Task 2 starts, the orthogonal loss would be enormous (Fast ≈ Slow → huge penalty) and destabilize training. Zero-init ensures only Fast learns Task 1.

---

## 3. Forward Pass

Every time the model processes input text, here is what happens inside each `DualLoRALinear`:

```
Input x  →  W_base @ x          (frozen base: does the heavy lifting)
         +  B_slow @ A_slow @ x  (slow branch: accumulated long-term memory)
         +  B_fast @ A_fast @ x  (fast branch: current task adaptation)
         =  y  (output)
```

**In code (dual_lora.py):**
```python
out  = F.linear(x, self.weight, self.bias)                      # base
slow = F.linear(F.linear(x, self.A_slow), self.B_slow) * scaling  # memory
fast = F.linear(F.linear(x, self.A_fast), self.B_fast) * scaling  # task
return out + slow + fast
```

**Visualized for one layer:**

```
                        ┌──── W_base (frozen) ────┐
                        │                         │
Input x ───────────────►│                         ├──► sum ──► y
                        │                         │     ▲
                        │ ┌── A_slow ──► B_slow ──┼─────┤
                        │ │  [8×1536]   [1536×8]  │     │
                        └─┤                       │   (scaling)
                          │ ┌── A_fast ──► B_fast ─┼─────┘
                          │ │  [8×1536]   [1536×8] │
                          └─┘                      │
                                                   │
```

The slow branch output is compressed: 1536 → 8 → 1536 dimensions (bottleneck of rank 8). This is very efficient — only 2 × (8 × 1536) = 24,576 parameters per layer instead of 1536² = 2,359,296.

### Classification Head

The final output of the last token goes into a `score` linear layer that maps to the total number of label classes (across all tasks). For 4 tasks: 2 + 4 + 2 + 14 = 22 total classes. The head is **trainable** in every task.

---

## 4. The Training Loop — Step by Step

### 4.1 What is Trainable at Each Task

| Task | Slow LoRA | Fast LoRA | Classification Head |
|------|-----------|-----------|---------------------|
| Task 1 | Frozen (zero, silent) | **Trainable** | **Trainable** |
| Task 2+ | **Frozen** (holds memory) | **Trainable** | **Trainable** |

Slow LoRA is **always frozen during training**. It is only updated once, at the task boundary, via consolidation.

### 4.2 Building Each Training Batch

For **Task 1**:
```
batch = [15 new Task-1 samples]
→ add to replay buffer (reservoir sampling)
```

For **Task 2+**:
```
new_batch  = [15 new Task-T samples]   ← from current task dataloader
replay     = [~3 random samples]       ← from replay buffer (all past tasks)
combined   = concat(new_batch, replay) ← total ~18 samples
→ also add new_batch to replay buffer
```

The `replay_ratio=0.2` means: `replay_n = int(15 × 0.2 / 0.8) = 3`.

### 4.3 Forward + Loss

```python
outputs = model(input_ids, attention_mask, labels=combined_labels)
task_loss = outputs.loss   # cross-entropy over 22 classes
```

For Task 2+, the soft orthogonality constraint is also added:
```python
L_orth = Σ_layers (||A_fast @ A_slow^T||²_F + ||B_fast^T @ B_slow||²_F)
total_loss = task_loss + λ × L_orth   # λ=0.01
```

`L_orth` penalizes row-space overlap between Fast and Slow matrices — if Fast's rows are drifting toward Slow's rows (into the "memory territory"), this loss pushes them apart.

### 4.4 Backward Pass — Gradients Flow Everywhere (Fast + Head)

```python
total_loss.backward()
```

After backward, gradients exist on:
- `A_fast.grad`, `B_fast.grad` for all 56 layers
- `score.weight.grad` (classification head)
- `A_slow.grad` = None (frozen → no gradient)
- `W_base.grad` = None (frozen → no gradient)

### 4.5 Orthogonal Projection (Hard Constraint)

This is the key step that makes DLOG different. **Before** the optimizer updates the weights, the Fast LoRA gradients are modified:

**Goal:** Ensure Fast LoRA does not learn in the directions already encoded by Slow LoRA.

**How:** Compute the orthonormal basis of Slow's "memory directions" via thin QR, then subtract those components from the gradient.

```
For each DualLoRALinear layer:

  A_fast gradient (shape [8, 1536]):
    Q_A, _ = QR(A_slow^T)           → [1536, 8] orthonormal basis of A_slow's row-space
    coeff  = grad_A @ Q_A           → [8, 8]  (how much gradient points toward memory)
    grad_A_proj = grad_A - coeff @ Q_A^T   → [8, 1536] (memory components removed)

  B_fast gradient (shape [1536, 8]):
    Q_B, _ = QR(B_slow)             → [1536, 8] orthonormal basis of B_slow's column-space
    coeff  = Q_B^T @ grad_B         → [8, 8]  (how much gradient points toward memory)
    grad_B_proj = grad_B - Q_B @ coeff     → [1536, 8] (memory components removed)
```

**Intuition:** Imagine A_slow's rows point in 8 specific directions in a 1536-dimensional space. The projection removes exactly those 8 directions from the gradient. Fast LoRA can still learn in the remaining 1528 directions — which is most of the space — but it cannot move in the directions that encode Slow's memory.

**Why QR, not the simpler formula?**
A naive approach would use `P = I - A_s A_s^T / ||A_s||^2`. But this operates in [r×r] (rank space), not in the [d×d] feature space. It only mixes/scales the 8 gradient rows — it does NOT remove any direction from R^{1536}. QR correctly builds the basis in feature space, making the projection meaningful.

**After projection:**
```python
nn.utils.clip_grad_norm_(params, max_grad_norm=1.0)
optimizer.step()
scheduler.step()
```

The optimizer now updates Fast LoRA using the *projected* gradients — guaranteed to be orthogonal to Slow's memory subspace.

---

## 5. Replay: How It Prevents Forgetting of Old Tasks

### 5.1 Reservoir Replay Buffer

The replay buffer is a fixed-size container (`buffer_size=750`) that stores samples from all past tasks. It uses **reservoir sampling**: as new data comes in, each new sample replaces a random existing one with probability `buffer_size / n_total_seen`.

Result: the buffer always contains a **uniform random sample** of everything seen so far.

**Buffer evolution:**
```
After Task 1 training:  [750 SST-2 samples]
After Task 2 training:  [~375 SST-2 | ~375 AG-News]
After Task 3 training:  [~250 SST-2 | ~250 AG-News | ~250 Amazon]
During Task 4 training: replay from above 750 mixed samples
```

### 5.2 What Replay Actually Does

During Task 4, each training step looks like:
```
[15 DBpedia samples] + [~3 random from {SST2, AGNews, Amazon}]
         ↓
   combined_batch (18 samples total)
         ↓
  forward → loss → backward → projection → optimizer.step()
```

The 3 replay samples provide gradient signal about old tasks. The optimizer nudges Fast LoRA (and the classification head) to also perform well on those old samples, not just DBpedia.

### 5.3 Replay + Projection: How They Interact

The projection and replay both work on the **same** backward pass. The combined_batch includes replay samples, so:

1. The loss includes errors from old-task replay samples
2. Gradients from those replay samples flow through Fast LoRA
3. Those gradients are then projected to be orthogonal to Slow's subspace

This means: the replay signal from old tasks is partially filtered by the projection. Replay gradients that point "toward" the Slow memory subspace get removed. This is acceptable because Slow already encodes that direction — the model doesn't need Fast to also move there.

### 5.4 IPC Scoring Microbatch

Every K=20 steps, a separate scoring-only pass runs:
```python
micro_batch = replay_buffer.sample(8)  # 8 pure replay samples
loss = model(micro_batch).loss
loss.backward()                        # compute gradients
ipc_tracker.update_from_grads()       # record gradient norms
optimizer.zero_grad()                  # DO NOT call optimizer.step()
```

This gives the importance tracker a clean signal about which modules are most sensitive to old-task data, without biasing toward the current task.

---

## 6. Task Boundary: Knowledge Transfer (Consolidation)

At the end of each task, **one** crucial operation happens: **Fast LoRA's knowledge is absorbed into Slow LoRA**, and Fast is reset to a clean slate.

### 6.1 Why Not Just Keep Fast?

If Fast kept accumulating across tasks without consolidation:
- Task 1: Fast learns SST-2 (stored in A_fast, B_fast)
- Task 2: Fast is trained further → overwrites Task 1 knowledge in Fast
- No protection for Task 1 knowledge

By moving Task 1's learning into Slow (which is then frozen during Task 2), that knowledge is permanently protected.

### 6.2 The Consolidation Math (QR + SVD)

The problem: merging Slow (rank-8) + Fast (rank-8) gives rank-16. We need to compress back to rank-8.

**Step 1: Concatenate**
```
B_cat = [B_slow | B_fast]  →  shape [1536, 16]
A_cat = [A_slow ; A_fast]  →  shape [16, 1536]
```

**Step 2: QR decomposition** (separates directions from magnitudes)
```
B_cat = Q_B @ R_B    Q_B: [1536, 16] orthonormal,  R_B: [16, 16]
A_cat^T = Q_A @ R_A  Q_A: [1536, 16] orthonormal,  R_A: [16, 16]
```

**Step 3: Small core matrix**
```
M = R_B @ R_A^T    →  shape [16, 16]   (tiny! fits easily in memory)
```

**Step 4: SVD on small core** (find best rank-8 approximation)
```
M ≈ U_8 @ Σ_8 @ V_8^T   (keep only top-8 singular values/vectors)
```

**Step 5: Reconstruct new Slow**
```
B_slow_new = Q_B @ U_8 @ sqrt(Σ_8)   →  [1536, 8]
A_slow_new = sqrt(Σ_8) @ V_8^T @ Q_A^T  →  [8, 1536]
```

The result is a rank-8 Slow LoRA that captures the most important directions from **both** old Slow (Task 1 memory) and Fast (Task 2 learning).

**Step 6: Reset Fast**
```
A_fast ← Kaiming uniform init
B_fast ← Zero init
```
Fast is now a blank canvas, ready to learn Task 3.

### 6.3 Visual Timeline

```
Task 1 Training:
  Slow = [0]         (silent)
  Fast = [learns SST-2]

Task 1 → Boundary:
  Slow ← consolidate(Slow=0, Fast=SST2) = SST2_compressed
  Fast ← reset to zero/Kaiming

Task 2 Training:
  Slow = [SST2 knowledge] (frozen)
  Fast = [learns AG-News] (projected away from SST2 directions)

Task 2 → Boundary:
  Slow ← consolidate(Slow=SST2, Fast=AGNews) = SST2+AGNews_compressed
  Fast ← reset

Task 3 Training:
  Slow = [SST2+AGNews knowledge] (frozen)
  Fast = [learns Amazon] (projected away from SST2+AGNews directions)

...and so on
```

---

## 7. IMC Freezing: Protecting the Most Critical Modules

### 7.1 Motivation

Even with orthogonal projection, not all 56 LoRA modules are equally important. Some layers' Fast LoRA might be extremely critical for retaining Task 1 knowledge (e.g., certain attention layers in the middle of the network). If those modules keep getting updated in future tasks, the projection might not fully protect them.

IMC identifies these critical modules and **locks them completely** for future tasks.

### 7.2 Importance Scoring

During each task's training, IMC tracks gradient activity for every Fast LoRA module:

```python
# After every loss.backward():
I_A = abs(A_fast.grad)        # gradient norm per element of A_fast
I_B = abs(B_fast.grad)        # gradient norm per element of B_fast

bar_I_A = β1 * bar_I_A + (1-β1) * I_A   # EMA smoothing, β1=0.85
bar_I_B = β1 * bar_I_B + (1-β1) * I_B

S_module = (mean(bar_I_A) + mean(bar_I_B)) / 2   # one score per module
```

**Why gradient norm (not weight × gradient)?**
B_fast is initialized to zero. So `|B * ∇B| = |0 × ∇B| = 0` for the entire first few steps. The EMA gets anchored at zero and never recovers. Gradient norm `|∇B|` doesn't have this problem — the gradient is non-zero even when the weight is zero.

### 7.3 Freeze Decision at Task Boundary

After consolidation at the end of each task:

```
1. Compute S_module for all 56 modules
2. Sort modules by score (highest = most important)
3. Select top 10% = top 5-6 modules → freeze them
4. Enforce global cap: at most 30% (≈17 modules) ever frozen cumulatively
```

**What does "frozen" mean in practice?**
- `A_fast.requires_grad = False`
- `B_fast.requires_grad = False`
- These parameters are excluded from the optimizer's param groups
- During consolidation, their Slow slot is NOT overwritten by the (reset) Fast

### 7.4 Visual of IMC Over Tasks

```
After Task 1:
  Scored: [L0.q: 0.003e-3, L0.v: 0.001e-3, ..., L14.q: 0.012e-3, ...]
  Freeze top 10%: {L14.q_proj, L20.v_proj, ...}  ← 5-6 modules

Task 2 Training:
  Frozen modules: skip (no gradient, no optimizer update)
  Others: train normally (with orthogonal projection)

After Task 2:
  Score non-frozen modules again
  Add more to freeze set (up to 30% total cap)

Task 3 Training:
  More modules frozen, even more protected
```

### 7.5 Consolidation + Freezing Interaction

```python
# At task boundary:
ipc_frozen = {keys of frozen modules}
model.consolidate_after_task(skip_keys=ipc_frozen)
```

For frozen modules: `reset_fast()` only — Slow is NOT touched.
For normal modules: full consolidation (Fast → Slow compressed, Fast reset).

This ensures the frozen modules' Slow LoRA keeps its original content untouched.

---

## 8. Full System Walkthrough: Task 2 Training Step

Here is everything that happens in a single training step during Task 2:

```
Step 1: Build batch
  new_samples = [15 AG-News items from dataloader]
  replay_samples = [3 random items from buffer (Task 1 SST-2)]
  combined = concat(new + replay)  [18 items total]
  → also add new_samples to replay buffer (reservoir sampling)

Step 2: Forward pass (for all 18 items)
  For each of 56 DualLoRALinear layers:
    y = W_base @ x  +  B_slow @ A_slow @ x  +  B_fast @ A_fast @ x
  Final output → classification head → logits [18, 22]

Step 3: Compute loss
  task_loss = CrossEntropy(logits, labels)
  L_orth = Σ ||A_fast @ A_slow^T||²_F + ||B_fast^T @ B_slow||²_F  (over 56 layers)
  total_loss = task_loss + 0.01 × L_orth

Step 4: Backward
  total_loss.backward()
  → A_fast.grad populated for all 56 layers (except frozen IMC modules)
  → B_fast.grad populated for all 56 layers (except frozen IMC modules)
  → score.weight.grad populated

Step 5: IPC stat update
  For each non-frozen module:
    bar_I_A ← EMA(|A_fast.grad|)
    bar_I_B ← EMA(|B_fast.grad|)

Step 6: Orthogonal projection (Hard Constraint)
  For each non-frozen, non-first-task DualLoRALinear:
    Q_A, _ = QR(A_slow^T)
    A_fast.grad ← A_fast.grad - (A_fast.grad @ Q_A) @ Q_A^T
    Q_B, _ = QR(B_slow)
    B_fast.grad ← B_fast.grad - Q_B @ (Q_B^T @ B_fast.grad)

Step 7: Gradient clipping + optimizer step
  clip_grad_norm_(params, max_norm=1.0)
  optimizer.step()    ← Fast LoRA and head are updated
  scheduler.step()

Step 8: (Every 20 steps) IPC scoring microbatch
  micro = replay_buffer.sample(8)
  micro_loss = model(micro).loss
  micro_loss.backward()
  bar_I ← update from pure replay gradients
  optimizer.zero_grad()   ← NO optimizer.step()

→ Repeat for 670 steps total
```

---

## 9. Summary Table

| Component | When | What it does |
|-----------|------|-------------|
| **Slow LoRA** | During training | Frozen — provides memory basis for projection |
| **Fast LoRA** | During training | Trainable — adapts to current task |
| **Orthogonal Projection** | Each backward step (Task 2+) | Removes Slow's directions from Fast's gradient |
| **Soft Constraint** | Each forward step (Task 2+) | Penalizes row-space overlap between Fast and Slow |
| **Replay** | Each training step (Task 2+) | Samples past-task data to keep head + Fast aligned |
| **IPC Scoring** | Each step + every 20 steps | Tracks gradient norm per module |
| **Consolidation** | Once per task boundary | Merges Fast into Slow via QR+SVD, resets Fast |
| **IMC Freeze** | Once per task boundary | Locks top-10% most important modules permanently |
| **Classification Head** | During training | Always trainable — adapts to all seen classes |
