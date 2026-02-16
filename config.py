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
    model_name: str = "t5-small"  # Switched from T5Gemma-2 for stability
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]  # Gemma-style attention names
    )

    # --- Orthogonal Gating ---
    lambda_orth: float = 0.1           # Soft constraint weight
    use_soft_constraint: bool = True
    use_hard_constraint: bool = True
    projection_type: str = "memory_gradient"  # "parameter" or "memory_gradient"
    memory_grad_buffer_size: int = 10  # K steps for incremental QR basis
    project_every_k: int = 1           # Project every K optimizer steps

    # --- Training ---
    learning_rate: float = 1e-4
    batch_size: int = 16  # t5-small is smaller, can use larger batch
    max_input_length: int = 256
    max_target_length: int = 64
    num_train_steps_per_task: int = 2000
    eval_every: int = 200
    log_every: int = 50
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # --- EMA (Slow LoRA consolidation) ---
    ema_decay: float = 0.999

    # --- Replay ---
    replay_buffer_size: int = 500       # per task
    replay_ratio: float = 0.2          # fraction of replay in each batch
    replay_strategy: str = "reservoir"  # "random" or "reservoir"

    # --- Vibe Coding Tasks (SST-2 Sentiment → AG News Topic) ---
    task_order: List[str] = field(
        default_factory=lambda: [
            "sst2", "ag_news"
        ]
    )

    # --- Paths ---
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

    # --- Smoke test ---
    smoke_test_steps: int = 5
    smoke_test_samples: int = 50

    # --- Device ---
    device: str = "cuda"
    fp16: bool = True

    # --- Seed ---
    seed: int = 42

    def __post_init__(self):
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
        "prompt_template": "sentiment: {text}",
    },
    "ag_news": {
        "hf_name": "ag_news",
        "hf_subset": None,
        "input_col": "text",
        "label_col": "label",
        "num_classes": 4,
        "class_names": ["World", "Sports", "Business", "Sci/Tech"],
        "prompt_template": "classify: {text}",
    },
    "amazon_reviews": {
        "hf_name": "amazon_polarity",
        "hf_subset": None,
        "input_col": "content",
        "label_col": "label",
        "num_classes": 2,
        "class_names": ["Negative", "Positive"],
        "prompt_template": "sentiment: {text}",
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
        "prompt_template": "topic: {text}",
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
        "prompt_template": "classify: {text}",
    },
}
