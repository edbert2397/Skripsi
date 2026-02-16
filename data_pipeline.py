"""
Data Pipeline for DLOG Continual Learning.

Handles:
  - Loading Standard CL benchmark datasets (AG News, Amazon, DBpedia, Yahoo)
  - Formatting for T5 (text-to-text)
  - Reservoir replay buffer
  - Mixed batch generation (new-task + replay)
"""
import random
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset, Dataset as HFDataset
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from config import DLOGConfig, DATASET_CONFIGS


# ======================================================================
# 1. Task Dataset (text-to-text for T5)
# ======================================================================
class CLTaskDataset(Dataset):
    """
    Wraps a HuggingFace dataset for a single CL task.
    Converts classification to text-to-text: "classify: {text}" -> "{label_name}"
    """

    def __init__(
        self,
        task_name: str,
        tokenizer: PreTrainedTokenizerBase,
        split: str = "train",
        max_input_length: int = 256,
        max_target_length: int = 16,
        max_samples: Optional[int] = None,
    ):
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        cfg = DATASET_CONFIGS[task_name]
        self.cfg = cfg

        # Load dataset
        if cfg["hf_subset"]:
            ds = load_dataset(cfg["hf_name"], cfg["hf_subset"], split=split,
                              trust_remote_code=True)
        else:
            ds = load_dataset(cfg["hf_name"], split=split,
                              trust_remote_code=True)

        if max_samples and len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))

        self.data = ds
        self.class_names = cfg["class_names"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        input_text = self.cfg["prompt_template"].format(
            text=item[self.cfg["input_col"]]
        )
        target_text = self.class_names[item[self.cfg["label_col"]]]

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        labels = targets.input_ids.squeeze(0)
        # Replace padding token id with -100 for loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ======================================================================
# 2. Reservoir Replay Buffer
# ======================================================================
class ReservoirReplayBuffer:
    """
    Reservoir sampling buffer for experience replay.

    Maintains a fixed-size buffer; when the buffer is full, each new sample
    replaces an existing sample with probability (buffer_size / n_seen).
    This guarantees a uniform random sample over all data seen so far.
    """

    def __init__(self, buffer_size: int = 500):
        self.buffer_size = buffer_size
        self.buffer: List[Dict[str, torch.Tensor]] = []
        self.n_seen = 0

    def add(self, sample: Dict[str, torch.Tensor]):
        """Add a sample via reservoir sampling."""
        self.n_seen += 1
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(sample)
        else:
            idx = random.randint(0, self.n_seen - 1)
            if idx < self.buffer_size:
                self.buffer[idx] = sample

    def add_batch(self, batch: Dict[str, torch.Tensor]):
        """Add all samples from a collated batch dict."""
        bsz = batch["input_ids"].shape[0]
        for i in range(bsz):
            sample = {k: v[i].cpu().clone() for k, v in batch.items()}
            self.add(sample)

    def sample(self, n: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample n items from the buffer and collate into a batch dict."""
        if len(self.buffer) == 0:
            return None
        n = min(n, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), n)
        samples = [self.buffer[i] for i in indices]
        return self._collate(samples)

    def _collate(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in samples[0]:
            batch[key] = torch.stack([s[key] for s in samples])
        return batch

    def __len__(self):
        return len(self.buffer)


# ======================================================================
# 3. Mixed DataLoader (Task + Replay)
# ======================================================================
def create_mixed_batch(
    task_batch: Dict[str, torch.Tensor],
    replay_buffer: ReservoirReplayBuffer,
    replay_ratio: float = 0.2,
    device: str = "cuda",
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """
    Create a mixed batch:  task_batch (full) + replay_batch (replay_ratio fraction).

    Returns (combined_batch, replay_batch_only)
    The replay_batch_only is needed for memory-gradient projection.
    """
    task_batch = {k: v.to(device) for k, v in task_batch.items()}
    bsz = task_batch["input_ids"].shape[0]
    replay_n = max(1, int(bsz * replay_ratio / (1.0 - replay_ratio)))

    replay_batch = replay_buffer.sample(replay_n)

    if replay_batch is None:
        return task_batch, None

    replay_batch = {k: v.to(device) for k, v in replay_batch.items()}

    # Concatenate
    combined = {}
    for key in task_batch:
        combined[key] = torch.cat([task_batch[key], replay_batch[key]], dim=0)

    return combined, replay_batch


# ======================================================================
# 4. Evaluation Dataset
# ======================================================================
class CLEvalDataset(Dataset):
    """
    Evaluation dataset — stores (input_text, label_index, class_names)
    for accuracy computation via generation.
    """

    def __init__(
        self,
        task_name: str,
        tokenizer: PreTrainedTokenizerBase,
        split: str = "test",
        max_input_length: int = 256,
        max_samples: Optional[int] = 500,
    ):
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

        cfg = DATASET_CONFIGS[task_name]
        self.cfg = cfg
        self.class_names = cfg["class_names"]

        if cfg["hf_subset"]:
            ds = load_dataset(cfg["hf_name"], cfg["hf_subset"], split=split,
                              trust_remote_code=True)
        else:
            ds = load_dataset(cfg["hf_name"], split=split,
                              trust_remote_code=True)

        if max_samples and len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = self.cfg["prompt_template"].format(
            text=item[self.cfg["input_col"]]
        )
        label_idx = item[self.cfg["label_col"]]

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "label_idx": label_idx,
            "label_text": self.class_names[label_idx],
        }


# ======================================================================
# 5. Convenience: build all dataloaders for a CL task sequence
# ======================================================================
def build_task_dataloaders(
    config: DLOGConfig,
    tokenizer: PreTrainedTokenizerBase,
    max_train_samples: Optional[int] = None,
    max_eval_samples: int = 500,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """
    Build train and eval DataLoaders for each task in config.task_order.

    Returns:
        train_loaders: {task_name: DataLoader}
        eval_loaders:  {task_name: DataLoader}
    """
    train_loaders = {}
    eval_loaders = {}

    for task_name in config.task_order:
        train_ds = CLTaskDataset(
            task_name, tokenizer, split="train",
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length,
            max_samples=max_train_samples,
        )
        eval_ds = CLEvalDataset(
            task_name, tokenizer, split="test",
            max_input_length=config.max_input_length,
            max_samples=max_eval_samples,
        )

        train_loaders[task_name] = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True,
            num_workers=0, drop_last=True,
        )
        eval_loaders[task_name] = DataLoader(
            eval_ds, batch_size=config.batch_size, shuffle=False, num_workers=0,
        )

    return train_loaders, eval_loaders
