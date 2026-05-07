"""
datasets.py - CSV-backed dataset loading and T5 tokenisation.

Reads from cl14_balanced_train_cleaned.csv / cl14_balanced_test_cleaned.csv
at the project root. Only `task_name`, `class_label`, and `text` columns are
used. No task prefix is prepended to the input text.
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

from src.data.prompts import format_prompt


# Some rows (long IMDB / Yelp / Amazon reviews) exceed the default 131_072 limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = PROJECT_ROOT / "cl14_balanced_train_cleaned.csv"
TEST_CSV = PROJECT_ROOT / "cl14_balanced_test_cleaned.csv"

_CSV_CACHE: Dict[Path, Dict[str, List[dict]]] = {}


def _load_csv_grouped(csv_path: Path) -> Dict[str, List[dict]]:
    """Read a CSV once and group rows by task_name. Result is cached per path."""
    if csv_path in _CSV_CACHE:
        return _CSV_CACHE[csv_path]
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    grouped: Dict[str, List[dict]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped.setdefault(row["task_name"], []).append({
                "text": row["text"],
                "class_label": row["class_label"],
            })
    _CSV_CACHE[csv_path] = grouped
    return grouped


def _balanced_take(rows: List[dict], n_total: int, seed: int) -> List[dict]:
    """Take ~n_total rows balanced across class_label, deterministic given seed."""
    rng = np.random.default_rng(seed)
    by_label: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by_label.setdefault(r["class_label"], []).append(i)

    labels = sorted(by_label.keys())
    n_per_class = max(1, n_total // len(labels))

    chosen: List[int] = []
    for lbl in labels:
        idxs = np.array(by_label[lbl])
        perm = rng.permutation(len(idxs))
        chosen.extend(idxs[perm[: min(n_per_class, len(idxs))]].tolist())

    chosen = rng.permutation(chosen).tolist()
    return [rows[i] for i in chosen]


def get_task_labels(task_name: str) -> List[str]:
    """Sorted list of `class_label` strings observed for a task in the train CSV."""
    grouped = _load_csv_grouped(TRAIN_CSV)
    if task_name not in grouped:
        raise KeyError(
            f"task_name '{task_name}' not found in {TRAIN_CSV.name}. "
            f"Available: {sorted(grouped.keys())}"
        )
    return sorted({r["class_label"] for r in grouped[task_name]})


class CLTaskDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def load_task(
    task_name: str,
    tokenizer: T5Tokenizer,
    n_train: int,
    n_test: int,
    batch_size: int = 8,
    max_input_len: int = 512,
    max_target_len: int = 8,
    seed: int = 42,
    num_workers: int = 0,
    balanced: bool = True,
) -> Tuple[List[dict], DataLoader, DataLoader]:
    """
    Load a task from the CL14 cleaned CSVs, tokenise it, and return
    (train_samples, train_loader, test_loader).

    - Inputs are the raw `text` column (no prefix).
    - Targets are the raw `class_label` column.
    - When balanced=True, n_train // n_classes (and n_test // n_classes) rows
      are drawn per class via numpy.random.default_rng(seed).
    """
    train_grouped = _load_csv_grouped(TRAIN_CSV)
    test_grouped = _load_csv_grouped(TEST_CSV)

    if task_name not in train_grouped:
        raise KeyError(
            f"task_name '{task_name}' not found in {TRAIN_CSV.name}. "
            f"Available: {sorted(train_grouped.keys())}"
        )
    if task_name not in test_grouped:
        raise KeyError(
            f"task_name '{task_name}' not found in {TEST_CSV.name}. "
            f"Available: {sorted(test_grouped.keys())}"
        )

    if balanced:
        train_rows = _balanced_take(train_grouped[task_name], n_train, seed)
        test_rows = _balanced_take(test_grouped[task_name], n_test, seed)
    else:
        rng = np.random.default_rng(seed)
        all_train = train_grouped[task_name]
        all_test = test_grouped[task_name]
        ti = rng.permutation(len(all_train))[: min(n_train, len(all_train))]
        te = rng.permutation(len(all_test))[: min(n_test, len(all_test))]
        train_rows = [all_train[i] for i in ti.tolist()]
        test_rows = [all_test[i] for i in te.tolist()]

    def tokenise(row: dict) -> dict:
        prompt = format_prompt(task_name, row["text"])
        model_inputs = tokenizer(
            prompt,
            max_length=max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = tokenizer(
            text_target=row["class_label"],
            max_length=max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_enc["input_ids"].squeeze(0)
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }

    train_samples = [tokenise(r) for r in train_rows]
    test_samples = [tokenise(r) for r in test_rows]

    train_ds = CLTaskDataset(train_samples)
    test_ds = CLTaskDataset(test_samples)

    pin = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_fn, num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        collate_fn=_collate_fn, num_workers=num_workers, pin_memory=pin,
    )

    return train_samples, train_loader, test_loader
