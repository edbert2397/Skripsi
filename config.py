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
    lora_rank: int = 8
    lora_alpha: float = 16.0
    target_modules: Optional[List[str]] = None

    # --- Orthogonal Gating ---
    lambda_orth: float = 0.01           # Soft constraint weight (reduced from 0.1)
    use_soft_constraint: bool = True
    use_hard_constraint: bool = True
    projection_type: str = "memory_gradient"  # "parameter" or "memory_gradient"
    memory_grad_buffer_size: int = 10  # K steps for incremental QR basis
    project_every_k: int = 5           # Project every K optimizer steps

    # --- Training ---
    learning_rate: float = 2e-4
    batch_size: int = 15                # Increased for speed
    replay_batch_size: int = 4         # Reduced for 6GB VRAM
    max_input_length: int = 256
    max_target_length: int = 64
    num_train_steps_per_task: int = 670
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

    # --- Seed ---
    seed: int = 42

    def __post_init__(self):
        # Auto-detect target modules if not specified
        if self.target_modules is None:
            name_lower = self.model_name.lower()
            if "t5gemma" in name_lower:
                self.target_modules = ["q_proj", "v_proj"]
            elif "gemma" in name_lower or "llama" in name_lower:
                self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            elif "t5" in name_lower:
                self.target_modules = ["q", "v"]
            else:
                self.target_modules = ["q_proj", "v_proj"]

        # Save results/checkpoints in model-specific subdirectories
        safe_model_name = self.model_name.replace("/", "_")
        self.output_dir = os.path.join(self.output_dir, safe_model_name)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, safe_model_name)

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
