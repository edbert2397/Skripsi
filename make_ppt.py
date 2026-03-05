"""Generate DLOG presentation PowerPoint."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Color Palette ──
BG_DARK   = RGBColor(0x1A, 0x1A, 0x2E)  # dark navy
BG_MID    = RGBColor(0x16, 0x21, 0x3E)  # slightly lighter
ACCENT    = RGBColor(0x00, 0xD2, 0xFF)  # cyan accent
ACCENT2   = RGBColor(0xFF, 0x6B, 0x6B)  # coral/red accent
ACCENT3   = RGBColor(0x4E, 0xCB, 0x71)  # green accent
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT     = RGBColor(0xCC, 0xCC, 0xCC)
YELLOW    = RGBColor(0xFF, 0xD9, 0x3D)
ORANGE    = RGBColor(0xFF, 0xA5, 0x00)
CODE_BG   = RGBColor(0x0D, 0x11, 0x17)  # very dark for code blocks

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SLIDE_W = prs.slide_width
SLIDE_H = prs.slide_height


# ── Helpers ──
def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_textbox(slide, left, top, width, height, text, font_size=18,
                bold=False, color=WHITE, align=PP_ALIGN.LEFT, font_name="Segoe UI"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = align
    return txBox

def add_bullet_frame(slide, left, top, width, height, items, font_size=18,
                     color=WHITE, spacing=Pt(8), font_name="Segoe UI"):
    """Add a textbox with multiple bullet-point paragraphs."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = font_name
        p.space_after = spacing
        p.level = 0
    return txBox

def add_code_block(slide, left, top, width, height, code_text, font_size=13):
    """Add a dark code block with monospace font."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = CODE_BG
    shape.line.fill.background()
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(12)
    tf.margin_top = Pt(8)
    tf.margin_right = Pt(12)
    tf.margin_bottom = Pt(8)
    lines = code_text.strip().split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(font_size)
        p.font.color.rgb = ACCENT
        p.font.name = "Consolas"
        p.space_after = Pt(2)
    return shape

def add_formula_box(slide, left, top, width, height, formula_text, font_size=16, color=YELLOW):
    """Add a formula in a subtle bordered box."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x10, 0x15, 0x25)
    shape.line.color.rgb = ACCENT
    shape.line.width = Pt(1.5)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(12)
    tf.margin_top = Pt(6)
    lines = formula_text.strip().split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Consolas"
        p.alignment = PP_ALIGN.CENTER
        p.space_after = Pt(4)
    return shape

def add_accent_line(slide, left, top, width):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Pt(3))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()

def add_section_number(slide, number, left=Inches(0.5), top=Inches(0.4)):
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, top, Inches(0.6), Inches(0.6))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()
    tf = shape.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = str(number)
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = BG_DARK
    p.font.name = "Segoe UI"
    p.alignment = PP_ALIGN.CENTER


# ====================================================================
# SLIDE 1 — Title
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(1), Inches(1.8), Inches(11), Inches(1.2),
            "DLOG", font_size=60, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1), Inches(3.0), Inches(11), Inches(1),
            "Dual-LoRA Orthogonal Gating\nfor Continual Learning on Large Language Models",
            font_size=28, bold=False, color=WHITE, align=PP_ALIGN.CENTER)
add_accent_line(slide, Inches(4.5), Inches(4.3), Inches(4))
add_textbox(slide, Inches(1), Inches(4.7), Inches(11), Inches(0.8),
            "Backbone: Qwen2.5-1.5B  |  Tasks: SST-2, AG News, Amazon Reviews, DBPedia-14",
            font_size=18, color=LIGHT, align=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1), Inches(5.8), Inches(11), Inches(0.6),
            "Skripsi Presentation — 2026",
            font_size=16, color=LIGHT, align=PP_ALIGN.CENTER)


# ====================================================================
# SLIDE 2 — Problem: Catastrophic Forgetting
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, "!")

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Problem: Catastrophic Forgetting", font_size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

add_bullet_frame(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(2.5), [
    "LLMs fine-tuned on new tasks forget previous knowledge",
    "Sequential learning: Task 1 → Task 2 → ... → Task N",
    "After learning Task N, performance on Tasks 1..N-1 drops",
    "Need: learn new tasks WITHOUT destroying old knowledge",
], font_size=20, color=LIGHT)

# Visual: forgetting diagram box
add_formula_box(slide, Inches(7), Inches(1.5), Inches(5.5), Inches(2),
    "After Task 1: SST-2  acc = 92%\n"
    "After Task 2: SST-2  acc = 55%  ← Forgetting!\n"
    "              AG News acc = 87%",
    font_size=16, color=ACCENT2)

add_textbox(slide, Inches(0.8), Inches(4.3), Inches(11.5), Inches(2.5),
            "DLOG Solution: 3 complementary mechanisms",
            font_size=26, bold=True, color=ACCENT)
add_bullet_frame(slide, Inches(0.8), Inches(5.1), Inches(11.5), Inches(2), [
    "1. Dual-LoRA Architecture — separate Slow (memory) and Fast (task) adapters",
    "2. Orthogonal Gating — soft + hard constraints keep Fast ⊥ Slow",
    "3. IPC Neuron Freezing — freeze the most important modules to protect memory",
    "  + Experience Replay — reservoir sampling from past tasks",
], font_size=18, color=LIGHT)


# ====================================================================
# SLIDE 3 — Model Backbone
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, 1)

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Model Backbone: Qwen2.5-1.5B", font_size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

add_bullet_frame(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(3), [
    "Base: Qwen2.5-1.5B (decoder-only LLM)",
    "Head: AutoModelForSequenceClassification",
    "Total labels = sum of all task classes (2+4+2+14 = 22)",
    "All base weights FROZEN (requires_grad=False)",
    "Only LoRA adapters + classification head are trainable",
    "Gradient checkpointing enabled to save VRAM",
], font_size=18, color=LIGHT)

# Architecture diagram as code block
add_code_block(slide, Inches(7), Inches(1.3), Inches(5.8), Inches(4),
    "Qwen2.5-1.5B (FROZEN)\n"
    "  ├── Embedding Layer\n"
    "  ├── Transformer Layers x28\n"
    "  │     ├── Self-Attention\n"
    "  │     │    ├── q_proj ← DualLoRA injected\n"
    "  │     │    ├── k_proj ← DualLoRA injected\n"
    "  │     │    ├── v_proj ← DualLoRA injected\n"
    "  │     │    └── o_proj ← DualLoRA injected\n"
    "  │     └── MLP (frozen)\n"
    "  └── Classification Head (TRAINABLE)\n"
    "       └── score: Linear(1536, 22)",
    font_size=15)

add_textbox(slide, Inches(0.8), Inches(5.0), Inches(5.5), Inches(0.5),
            "LoRA Config:", font_size=20, bold=True, color=ACCENT)
add_bullet_frame(slide, Inches(0.8), Inches(5.5), Inches(11), Inches(1.5), [
    "rank = 16,  alpha = 16,  scaling = alpha/rank = 1.0",
    "Target modules: q_proj, k_proj, v_proj, o_proj  (4 per layer × 28 layers = 112 DualLoRA modules)",
], font_size=17, color=LIGHT)


# ====================================================================
# SLIDE 4 — Dual-LoRA: Slow + Fast Architecture
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, 2)

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Dual-LoRA: Slow (Memory) + Fast (Task)", font_size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

# Left column — Slow
add_textbox(slide, Inches(0.8), Inches(1.5), Inches(5), Inches(0.5),
            "Slow LoRA (Memory)", font_size=24, bold=True, color=ACCENT)
add_bullet_frame(slide, Inches(0.8), Inches(2.1), Inches(5.5), Inches(2.5), [
    "Stores consolidated knowledge from ALL past tasks",
    "ZERO-initialized → silent during Task 1",
    "NEVER trained directly (requires_grad=False)",
    "Only updated at task boundary via consolidation",
    "Provides stable null-space for orthogonal projection",
], font_size=17, color=LIGHT)

# Right column — Fast
add_textbox(slide, Inches(7), Inches(1.5), Inches(5), Inches(0.5),
            "Fast LoRA (Task)", font_size=24, bold=True, color=ACCENT2)
add_bullet_frame(slide, Inches(7), Inches(2.1), Inches(5.5), Inches(2.5), [
    "Learns the CURRENT task only",
    "Kaiming-init A_fast, Zero-init B_fast",
    "Trained via backprop every step",
    "RESET after each task consolidation",
    "Constrained to be orthogonal to Slow",
], font_size=17, color=LIGHT)

# Forward formula
add_formula_box(slide, Inches(1.5), Inches(4.7), Inches(10), Inches(1.2),
    "y = W_frozen · x  +  (α/r) · B_slow @ A_slow @ x  +  (α/r) · B_fast @ A_fast @ x\n"
    "     ──────────       ─────────────────────────────    ─────────────────────────────\n"
    "     Base (frozen)          Slow (Memory)                    Fast (Task)",
    font_size=16, color=YELLOW)

# Init detail
add_code_block(slide, Inches(1.5), Inches(6.1), Inches(10), Inches(1.1),
    "# Initialization\n"
    "Slow: A_slow = 0, B_slow = 0          # Silent until consolidation\n"
    "Fast: A_fast = Kaiming, B_fast = 0    # Standard LoRA init",
    font_size=14)


# ====================================================================
# SLIDE 5 — Forward Pass Flow
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, 3)

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Forward Pass: How DualLoRALinear Works", font_size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

add_code_block(slide, Inches(0.8), Inches(1.5), Inches(11.5), Inches(3.2),
    "def forward(self, x):                          # x: [batch, seq_len, d_in]\n"
    "    # 1. Base frozen linear\n"
    "    out = F.linear(x, W_frozen, bias)          # [batch, seq_len, d_out]\n"
    "\n"
    "    # 2. Slow LoRA (Memory branch) — encodes knowledge from tasks 1..t-1\n"
    "    slow = F.linear(F.linear(x, A_slow), B_slow) * (α/r)\n"
    "    #      x → [b,s,d_in] @ A_slow^T → [b,s,r] @ B_slow^T → [b,s,d_out]\n"
    "\n"
    "    # 3. Fast LoRA (Task branch) — learns current task t\n"
    "    fast = F.linear(F.linear(x, A_fast), B_fast) * (α/r)\n"
    "\n"
    "    return out + slow + fast                    # Sum of all 3 branches",
    font_size=15)

add_textbox(slide, Inches(0.8), Inches(5.0), Inches(11.5), Inches(0.5),
            "Key Insight:", font_size=22, bold=True, color=ACCENT)
add_bullet_frame(slide, Inches(0.8), Inches(5.5), Inches(11.5), Inches(1.5), [
    "Task 1: Slow = 0 (zero-init), so y = W·x + Fast — only Fast learns",
    "Task 2+: Slow = consolidated memory (frozen), Fast learns in null-space of Slow",
    "Deployed: After consolidation, Fast is merged into Slow → y = W·x + Slow_new",
], font_size=18, color=LIGHT)


# ====================================================================
# SLIDE 6 — Forward Transfer & Backward Transfer
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, 4)

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Forward & Backward Transfer", font_size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

# Forward Transfer
add_textbox(slide, Inches(0.8), Inches(1.4), Inches(5.5), Inches(0.5),
            "Forward Transfer (new tasks benefit from old)", font_size=22, bold=True, color=ACCENT3)
add_bullet_frame(slide, Inches(0.8), Inches(2.0), Inches(5.5), Inches(2.5), [
    "Slow LoRA provides a \"knowledge backbone\"",
    "New task starts with: y = W·x + Slow_memory",
    "Fast learns complementary features in null-space",
    "Shared classification head accumulates all classes",
    "Replay mixes old + new data → shared representations",
], font_size=17, color=LIGHT)

# Backward Transfer
add_textbox(slide, Inches(7), Inches(1.4), Inches(5.5), Inches(0.5),
            "Backward Transfer (protect old tasks)", font_size=22, bold=True, color=ACCENT2)
add_bullet_frame(slide, Inches(7), Inches(2.0), Inches(5.5), Inches(2.5), [
    "3 mechanisms prevent catastrophic forgetting:",
    "  (a) Orthogonal Gating: Fast ⊥ Slow",
    "  (b) Experience Replay: rehearse old data",
    "  (c) IPC Freezing: lock critical modules",
    "Slow is NEVER modified during training",
], font_size=17, color=LIGHT)

# Visual flow
add_formula_box(slide, Inches(0.8), Inches(4.7), Inches(11.5), Inches(2.2),
    "Task 1:  Train Fast₁  →  Consolidate: Slow = Fast₁,  Reset Fast\n"
    "Task 2:  Train Fast₂ ⊥ Slow  →  Consolidate: Slow = merge(Slow, Fast₂),  Reset Fast\n"
    "Task 3:  Train Fast₃ ⊥ Slow  →  Consolidate: Slow = merge(Slow, Fast₃),  Reset Fast\n"
    "                        ↑                         ↑\n"
    "             orthogonal constraint        QR+SVD rank-r compression",
    font_size=16, color=YELLOW)


# ====================================================================
# SLIDE 7 — Replay: Reservoir Sampling
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, 5)

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Mechanism 1: Experience Replay", font_size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

add_bullet_frame(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(2.5), [
    "Reservoir Sampling — uniform coverage of ALL past tasks",
    "Buffer size: 400 samples per task",
    "Replay ratio: 10% of each training batch",
    "Samples added from current task too (for future replay)",
    "Mixed batch = (1−ρ) · current_data + ρ · replay_data",
], font_size=18, color=LIGHT)

# Formula
add_formula_box(slide, Inches(7), Inches(1.5), Inches(5.5), Inches(2.5),
    "Reservoir Sampling (Vitter, 1985)\n\n"
    "For i-th incoming sample:\n"
    "  if buffer not full: add sample\n"
    "  else:\n"
    "    j = random(0, i)\n"
    "    if j < buffer_size: replace buffer[j]",
    font_size=15, color=YELLOW)

add_code_block(slide, Inches(0.8), Inches(4.3), Inches(11.5), Inches(2.8),
    "# Training step with replay (Task 2+)\n"
    "combined_batch, replay_batch = create_mixed_batch(\n"
    "    current_batch,\n"
    "    replay_buffer,\n"
    "    replay_ratio=0.1,    # 10% replay, 90% current task\n"
    ")\n"
    "replay_buffer.add_batch(current_batch)  # Add to buffer for future tasks\n"
    "\n"
    "loss = model(combined_batch)            # Train on mixed data\n"
    "# Both current task AND old tasks contribute to gradients",
    font_size=14)


# ====================================================================
# SLIDE 8 — Orthogonal Gating: Soft Constraint
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, 6)

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Mechanism 2a: Soft Orthogonal Constraint", font_size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

add_textbox(slide, Inches(0.8), Inches(1.4), Inches(11), Inches(0.6),
            "Penalize overlap between Fast and Slow subspaces via regularization loss",
            font_size=20, color=LIGHT)

add_formula_box(slide, Inches(1.5), Inches(2.2), Inches(10), Inches(1.5),
    "L_orth  =  Σᵢ  ( ||A_fast_i  ·  A_slow_iᵀ||²_F   +   ||B_fast_iᵀ  ·  B_slow_i||²_F )\n\n"
    "L_total = L_task  +  λ_orth · L_orth        (λ_orth = 0.01)",
    font_size=18, color=YELLOW)

add_code_block(slide, Inches(0.8), Inches(4.0), Inches(11.5), Inches(2.8),
    "def compute_orth_loss(dual_layers):\n"
    "    loss = 0\n"
    "    for layer in dual_layers:\n"
    "        A_s, B_s = layer.get_slow_params()    # [r, d_in], [d_out, r]\n"
    "        A_f, B_f = layer.get_fast_params()\n"
    "\n"
    "        # Penalize row-space overlap:  A_f · A_sᵀ should be zero-matrix\n"
    "        loss += ||A_f @ A_s.T||²_F             # [r, r] → scalar\n"
    "        # Penalize column-space overlap: B_fᵀ · B_s should be zero-matrix\n"
    "        loss += ||B_f.T @ B_s||²_F             # [r, r] → scalar\n"
    "    return loss",
    font_size=14)

add_textbox(slide, Inches(0.8), Inches(6.9), Inches(11), Inches(0.5),
            "Intuition: If A_fast ⊥ A_slow, they span different directions → new task doesn't interfere with memory",
            font_size=17, color=ACCENT)


# ====================================================================
# SLIDE 9 — Orthogonal Gating: Hard Constraint (GPM-style)
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, 7)

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Mechanism 2b: Hard Gradient Projection (GPM-style)", font_size=34, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

add_textbox(slide, Inches(0.8), Inches(1.4), Inches(11), Inches(0.6),
            "After backward: project Fast gradients into the null-space of Slow's subspace",
            font_size=20, color=LIGHT)

# Formula for A
add_formula_box(slide, Inches(0.8), Inches(2.2), Inches(5.5), Inches(1.8),
    "For A_fast (shape [r, d_in]):\n"
    "  Q_A, _ = QR(A_slowᵀ)     ← orthonormal basis\n"
    "  g_proj = g − (g · Q_A) · Q_Aᵀ\n"
    "  → removes component in A_slow's row-space",
    font_size=15, color=YELLOW)

# Formula for B
add_formula_box(slide, Inches(7), Inches(2.2), Inches(5.5), Inches(1.8),
    "For B_fast (shape [d_out, r]):\n"
    "  Q_B, _ = QR(B_slow)      ← orthonormal basis\n"
    "  g_proj = g − Q_B · (Q_Bᵀ · g)\n"
    "  → removes component in B_slow's col-space",
    font_size=15, color=YELLOW)

add_code_block(slide, Inches(0.8), Inches(4.3), Inches(11.5), Inches(2.8),
    "def project_gradients_parameter(dual_layers):\n"
    "    for layer in dual_layers:\n"
    "        A_s, B_s = layer.get_slow_params()\n"
    "        A_f, B_f = layer.get_fast_params()\n"
    "\n"
    "        # A_fast: null-space of A_slow's row-space (in R^{d_in})\n"
    "        Q_A, _ = torch.linalg.qr(A_s.T)       # [d_in, r]\n"
    "        A_f.grad = g - (g @ Q_A) @ Q_A.T       # project out Slow's directions\n"
    "\n"
    "        # B_fast: null-space of B_slow's column-space (in R^{d_out})\n"
    "        Q_B, _ = torch.linalg.qr(B_s)          # [d_out, r]\n"
    "        B_f.grad = g - Q_B @ (Q_B.T @ g)       # project out Slow's directions",
    font_size=13)

add_textbox(slide, Inches(0.8), Inches(7.1), Inches(11), Inches(0.4),
            "Ref: Saha et al., \"Gradient Projection Memory\", ICLR 2021 — adapted for LoRA parameters",
            font_size=14, color=LIGHT)


# ====================================================================
# SLIDE 10 — IPC Neuron Freezing
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, 8)

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Mechanism 3: IPC Important Module Freezing", font_size=34, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

add_textbox(slide, Inches(0.8), Inches(1.3), Inches(11), Inches(0.5),
            "Freeze the most \"important\" LoRA modules to permanently protect critical memory",
            font_size=20, color=LIGHT)

add_formula_box(slide, Inches(0.8), Inches(2.0), Inches(5.5), Inches(3.5),
    "Step 1: Sensitivity (per step)\n"
    "  I(θ) = |∇θ|          (gradient magnitude)\n\n"
    "Step 2: EMA Smoothing\n"
    "  bar_I ← β₁·bar_I + (1−β₁)·I    (β₁=0.85)\n\n"
    "Step 3: Module Score\n"
    "  S(module) = mean(bar_I)  over A_fast, B_fast\n\n"
    "Step 4: Freeze top-p% (p=10%, cap=30%)",
    font_size=15, color=YELLOW)

add_code_block(slide, Inches(7), Inches(2.0), Inches(5.5), Inches(3.5),
    "# At each task boundary:\n"
    "scores = tracker.get_module_scores()\n"
    "ranked = sort(scores, descending)\n"
    "\n"
    "# Freeze top 10% of 112 modules\n"
    "#  ≈ 11 modules per task boundary\n"
    "# Global cap: max 30% ever frozen\n"
    "#  ≈ max 33 modules total\n"
    "\n"
    "for key in ranked[:top_p]:\n"
    "  A_fast.requires_grad = False\n"
    "  B_fast.requires_grad = False\n"
    "  # Skip in consolidate_after_task()",
    font_size=14)

add_bullet_frame(slide, Inches(0.8), Inches(5.8), Inches(11.5), Inches(1.5), [
    "Why |∇θ| instead of |θ·∇θ|?  B_fast starts at 0 (LoRA init), so |θ·∇θ| = 0 always — scores collapse to zero",
    "Frozen modules: excluded from optimizer, Slow slot preserved during consolidation",
    "Effect: critical past-task parameters are permanently locked, preventing overwrite",
], font_size=16, color=LIGHT)


# ====================================================================
# SLIDE 11 — Task Consolidation (QR + SVD)
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, 9)

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Task Consolidation: QR + SVD Compression", font_size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

add_textbox(slide, Inches(0.8), Inches(1.4), Inches(11), Inches(0.5),
            "At end of each task: merge Slow + Fast back into rank-r Slow adapter",
            font_size=20, color=LIGHT)

add_formula_box(slide, Inches(0.8), Inches(2.1), Inches(11.5), Inches(3.3),
    "1. Concatenate:   B_cat = [B_slow, B_fast]   (d_out × 2r)\n"
    "                  A_cat = [A_slow; A_fast]   (2r × d_in)\n\n"
    "2. Thin QR:       Q_B, R_B = QR(B_cat)       Q_A, R_A = QR(A_catᵀ)\n\n"
    "3. Core matrix:   M = R_B · R_Aᵀ             (small 2r × 2r matrix)\n\n"
    "4. Best rank-r:   U, S, Vᵀ = SVD(M)          truncate to top-r singular values\n\n"
    "5. New Slow:      B_slow_new = Q_B · U_r · √S_r\n"
    "                  A_slow_new = √S_r · V_rᵀ · Q_Aᵀ",
    font_size=16, color=YELLOW)

add_code_block(slide, Inches(0.8), Inches(5.7), Inches(11.5), Inches(1.5),
    "# After consolidation:\n"
    "layer.A_slow = A_slow_new     # Updated memory (rank r preserved)\n"
    "layer.B_slow = B_slow_new\n"
    "layer.A_fast = Kaiming_init   # Reset for next task\n"
    "layer.B_fast = zeros          # Clean slate",
    font_size=14)


# ====================================================================
# SLIDE 12 — Full Training Pipeline
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, 10)

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Full Training Pipeline", font_size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

add_code_block(slide, Inches(0.5), Inches(1.3), Inches(12.3), Inches(5.8),
    "for task_idx, task in enumerate([SST-2, AG_News, Amazon_Reviews, DBPedia_14]):\n"
    "\n"
    "    # ── Phase: Training ──────────────────────────────────────────\n"
    "    freeze(Slow_LoRA)                 # Slow is read-only during training\n"
    "    optimizer = AdamW(Fast_params + classification_head)\n"
    "\n"
    "    for step in range(670):           # num_train_steps_per_task\n"
    "        batch = mix(current_data, replay_buffer, ratio=0.1)  # Step 1: Replay\n"
    "\n"
    "        loss = model(batch)           # Step 2: Forward (base + slow + fast)\n"
    "        L = loss + λ · L_orth         # Step 3: Soft orthogonal penalty\n"
    "        L.backward()                  # Step 4: Backward\n"
    "\n"
    "        ipc_tracker.update(grads)     # Step 5: Update importance stats\n"
    "        project_gradients(Fast, Slow) # Step 6: Hard gradient projection (GPM)\n"
    "        optimizer.step()              # Step 7: Update Fast only\n"
    "\n"
    "    # ── Phase: Task Boundary ─────────────────────────────────────\n"
    "    consolidate_after_task()          # QR+SVD: merge Fast → Slow, reset Fast\n"
    "    ipc_freeze(top_10%)              # Freeze most important modules\n"
    "    evaluate(all_tasks)              # Track CL metrics (ACC, BWT, FWT)",
    font_size=14)


# ====================================================================
# SLIDE 13 — Experiment Setup
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)
add_section_number(slide, 11)

add_textbox(slide, Inches(1.3), Inches(0.35), Inches(10), Inches(0.8),
            "Experiment Setup", font_size=36, bold=True, color=WHITE)
add_accent_line(slide, Inches(1.3), Inches(1.15), Inches(5))

# Table-like layout
items_left = [
    "Model: Qwen2.5-1.5B",
    "Tasks: SST-2 → AG News → Amazon → DBPedia-14",
    "Training steps: 670 per task",
    "Learning rate: 2e-4 (AdamW + linear warmup)",
    "Batch size: 15 (+ 4 replay)",
    "Max input length: 256 tokens",
]
items_right = [
    "LoRA rank: 16,  alpha: 16",
    "Replay buffer: 400 per task (reservoir)",
    "Replay ratio: 10%",
    "Soft constraint: λ_orth = 0.01",
    "Hard constraint: GPM parameter projection",
    "IPC: freeze top 10%, cap 30%",
]

add_textbox(slide, Inches(0.8), Inches(1.4), Inches(5.5), Inches(0.5),
            "Training Configuration", font_size=22, bold=True, color=ACCENT)
add_bullet_frame(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(3), items_left,
                 font_size=17, color=LIGHT)

add_textbox(slide, Inches(7), Inches(1.4), Inches(5.5), Inches(0.5),
            "DLOG-Specific Hyperparams", font_size=22, bold=True, color=ACCENT)
add_bullet_frame(slide, Inches(7), Inches(1.9), Inches(5.5), Inches(3), items_right,
                 font_size=17, color=LIGHT)

# Comparison
add_textbox(slide, Inches(0.8), Inches(5.0), Inches(11), Inches(0.5),
            "Baselines for Comparison:", font_size=22, bold=True, color=ACCENT2)
add_bullet_frame(slide, Inches(0.8), Inches(5.5), Inches(11.5), Inches(1.5), [
    "Single LoRA + Replay — no Slow/Fast separation, no orthogonal gating",
    "DLOG (no replay) — ablation: same architecture but pure sequential fine-tuning",
    "DLOG (full) — complete system with all 3 mechanisms",
], font_size=17, color=LIGHT)


# ====================================================================
# SLIDE 14 — Summary & Thank You
# ====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide, BG_DARK)

add_textbox(slide, Inches(1), Inches(1.2), Inches(11), Inches(1),
            "Summary", font_size=40, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
add_accent_line(slide, Inches(4.5), Inches(2.2), Inches(4))

add_bullet_frame(slide, Inches(1.5), Inches(2.6), Inches(10), Inches(3.5), [
    "DLOG separates knowledge into Slow (memory) and Fast (task) LoRA adapters",
    "Orthogonal Gating (soft + hard) ensures new learning does not corrupt old knowledge",
    "Reservoir Replay provides gradient signal from past tasks",
    "IPC Freezing permanently protects the most critical modules",
    "QR+SVD consolidation compresses both adapters back to rank-r at each task boundary",
    "All mechanisms are complementary — each addresses a different aspect of forgetting",
], font_size=19, color=LIGHT, spacing=Pt(14))

add_textbox(slide, Inches(1), Inches(6.2), Inches(11), Inches(0.8),
            "Thank You — Questions?",
            font_size=32, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


# ── Save ──
output_path = r"E:\Kuliah\sem8\skripsi\dlog\DLOG_Presentation.pptx"
prs.save(output_path)
print(f"Saved to {output_path}")
