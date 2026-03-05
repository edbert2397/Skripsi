"""
DLOG Configuration — all hyperparameters and paths.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import os
import torch


@dataclass
class DLOGConfig:
    # --- Model ---
    model_name: str = "Qwen/Qwen2.5-1.5B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = False  # Disabled: GradScaler conflicts with multi-backward training loop
    lora_rank: int = 16
    lora_alpha: float = 16
    target_modules: Optional[List[str]] = None

    # --- Orthogonal Gating ---
    # lambda_orth is derived automatically in __post_init__ when auto_lambda_orth=True:
    #   0.01 when use_hard_constraint=True  (hard projection already handles most of the work)
    #   0.10 when use_hard_constraint=False (soft penalty must carry the full load)
    lambda_orth: float = 0.01           # placeholder; overwritten by __post_init__ if auto_lambda_orth=True
    auto_lambda_orth: bool = True       # if True, derive lambda_orth from use_hard_constraint
    use_soft_constraint: bool = True
    use_hard_constraint: bool = True
    projection_type: str = "parameter"  # "parameter" or "memory_gradient"
    memory_grad_buffer_size: int = 10  # K steps for incremental QR basis
    project_every_k: int = 1           # Project every K optimizer steps (k=1 is most correct
                                        # but doubles GPU memory; keep 5 for 6GB VRAM budget)

    # --- Training ---
    learning_rate: float = 2e-4
    batch_size: int = 15                # Increased for speed
    replay_batch_size: int = 4         # Reduced for 6GB VRAM
    max_input_length: int = 256
    max_target_length: int = 64
    num_train_steps_per_task: int = 1000
    eval_every: int = 200
    log_every: int = 50
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # --- Consolidation ---
    ema_decay: float = 0.99
    # Slow LoRA is ONLY updated at the END of each task via consolidate_after_task().
    # No EMA during training — this keeps P_slow = I - (A_s A_s^T)/(||A_s||^2 + λ)
    # constant throughout the task, ensuring a stable, fixed null-space projection.

    # --- Replay ---
    no_replay: bool = False            # if True, disable replay for ALL methods (pure sequential fine-tuning)
    replay_buffer_size: int = 400       # per task
    replay_ratio: float = 0.1          # fraction of replay in each batch
    replay_strategy: str = "reservoir"  # "random" or "reservoir"

    # --- Vibe Coding Tasks (SST-2 Sentiment → AG News Topic) ---
    task_order: List[str] = field(
        default_factory=lambda: [
            "sst2",
            "ag_news",
            "amazon_reviews",
            "dbpedia_14"
        ]
    )

    # --- Paths ---
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

    # --- Smoke test ---
    smoke_test_steps: int = 10
    smoke_test_samples: int = 50

    # --- IPC: Important Module Freezing (arXiv:2504.13407 adaptation) ---
    # Freezes the most "important" LoRA (layer × target) modules for future tasks
    # based on gradient-sensitivity × uncertainty scores.
    ipc_enabled: bool = True
    ipc_beta1: float = 0.85                    # EMA decay for sensitivity bar_I
    ipc_beta2: float = 0.85                    # EMA decay for uncertainty bar_U
    ipc_freeze_p: float = 0.10                 # fraction of modules to freeze per task boundary
    ipc_max_frozen_fraction: float = 0.30      # global cap: at most 30% of modules ever frozen
    ipc_update_every_n_steps: int = 1          # how often (in steps) to update stats from normal grads
    ipc_scoring_microbatch_enabled: bool = True
    ipc_scoring_microbatch_every_k_steps: int = 20   # replay-only scoring pass every K steps
    ipc_scoring_microbatch_bsz: int = 8        # size of scoring-only replay microbatch

    # --- Seed ---
    seed: int = 42

    def __post_init__(self):
        # No-replay mode: disable IPC freezing too (pure sequential fine-tuning)
        if self.no_replay:
            self.ipc_enabled = False

        # Auto-derive lambda_orth from constraint mode
        if self.auto_lambda_orth:
            self.lambda_orth = 0.01 if self.use_hard_constraint else 0.1

        # Auto-detect target modules if not specified
        if self.target_modules is None:
            name_lower = self.model_name.lower()
            if "t5gemma" in name_lower:
                self.target_modules = ["q_proj", "v_proj"]
            elif "gemma" in name_lower or "llama" in name_lower:
                self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            elif "qwen" in name_lower:
                self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            elif "t5" in name_lower:
                self.target_modules = ["q", "v"]
            else:
                self.target_modules = ["q_proj", "v_proj"]

        # Save results/checkpoints in model-specific subdirectories,
        # with a replay-mode suffix so runs never overwrite each other.
        safe_model_name = self.model_name.replace("/", "_")
        replay_suffix = "no-replay" if self.no_replay else "replay"
        self.output_dir = os.path.join(self.output_dir, safe_model_name, replay_suffix)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, safe_model_name, replay_suffix)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


# Dataset configs: HuggingFace dataset name -> (input_col, target_col, num_classes)
DATASET_CONFIGS = {
    "sst2": {
        "hf_name": "glue",
        "hf_subset": "sst2",
        "input_col": "sentence",
        "label_col": "label",
        "num_classes": 2,
        "class_names": ["Negative", "Positive"],
        # Sequence Classification prompt: no prompt template needed, just text!
        "prompt_template": "Classify the sentiment of this review as Positive or Negative:\nReview: {text}",
        "eval_split": "validation",  # GLUE SST-2 has no labeled test split
    },
    "ag_news": {
        "hf_name": "ag_news",
        "hf_subset": None,
        "input_col": "text",
        "label_col": "label",
        "num_classes": 4,
        "class_names": ["World", "Sports", "Business", "Sci/Tech"],
        "prompt_template": "Classify the topic of this news article as World, Sports, Business, or Sci/Tech:\nArticle: {text}",
    },
    "amazon_reviews": {
        "hf_name": "amazon_polarity",
        "hf_subset": None,
        "input_col": "content",
        "label_col": "label",
        "num_classes": 2,
        "class_names": ["Negative", "Positive"],
        "prompt_template": "Classify the sentiment of this review as Positive or Negative:\nReview: {text}\nSentiment:",
    },
    "dbpedia_14": {
        "hf_name": "fancyzhx/dbpedia_14",
        "hf_subset": None,
        "input_col": "content",
        "label_col": "label",
        "num_classes": 14,
        "class_names": [
            "Company", "EducationalInstitution", "Artist", "Athlete",
            "OfficeHolder", "MeanOfTransportation", "Building",
            "NaturalPlace", "Village", "Animal", "Plant", "Album",
            "Film", "WrittenWork"
        ],
        "prompt_template": "Classify the topic of this article into one of the following categories: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork:\nArticle: {text}\nTopic:",
    },
    "yahoo_answers": {
        "hf_name": "yahoo_answers_topics",
        "hf_subset": None,
        "input_col": "question_title",
        "label_col": "topic",
        "num_classes": 10,
        "class_names": [
            "Society & Culture", "Science & Mathematics", "Health",
            "Education & Reference", "Computers & Internet", "Sports",
            "Business & Finance", "Entertainment & Music",
            "Family & Relationships", "Politics & Government"
        ],
        "prompt_template": "Classify the topic of this question into one of the following categories: Society & Culture, Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships, Politics & Government:\nQuestion: {text}\nTopic:",
    },
}
