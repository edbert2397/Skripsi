"""
datasets.py - Dataset loading and T5 tokenisation for all benchmark tasks.

All tasks are framed as seq2seq: input is "task: text", output is label string.
This matches the SuRe paper's T5-Large setup.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from datasets import load_dataset
from typing import List, Tuple, Dict, Optional


# ---- Task definitions ----
TASK_CONFIG = {
    "ag_news": {
        "hf_name": "ag_news",
        "text_key": "text",
        "label_key": "label",
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "prefix": "classify news: ",
    },
    "amazon_reviews": {
        "hf_name": "amazon_polarity",
        "text_key": "content",
        "label_key": "label",
        "labels": ["negative", "positive"],
        "prefix": "classify sentiment: ",
    },
    "dbpedia": {
        "hf_name": "dbpedia_14",
        "text_key": "content",
        "label_key": "label",
        "labels": [
            "Company", "EducationalInstitution", "Artist", "Athlete",
            "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
            "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
        ],
        "prefix": "classify topic: ",
    },
    "yahoo_answers": {
        "hf_name": "yahoo_answers_topics",
        "text_key": "question_title",
        "label_key": "topic",
        "labels": [
            "Society & Culture", "Science & Mathematics", "Health",
            "Education & Reference", "Computers & Internet", "Sports",
            "Business & Finance", "Entertainment & Music",
            "Family & Relationships", "Politics & Government"
        ],
        "prefix": "classify question: ",
    },
    "sst2": {
        "hf_name": ("glue", "sst2"),
        "text_key": "sentence",
        "label_key": "label",
        "labels": ["negative", "positive"],
        "prefix": "classify sentiment: ",
    },
    "mnli": {
        "hf_name": ("glue", "mnli"),
        "text_key": None,   # special handling
        "label_key": "label",
        "labels": ["entailment", "neutral", "contradiction"],
        "prefix": "classify nli: ",
        "test_split": "validation_matched",
    },
    "qqp": {
        "hf_name": ("glue", "qqp"),
        "text_key": None,
        "label_key": "label",
        "labels": ["not duplicate", "duplicate"],
        "prefix": "classify paraphrase: ",
    },
    "rte": {
        "hf_name": ("glue", "rte"),
        "text_key": None,
        "label_key": "label",
        "labels": ["entailment", "not entailment"],
        "prefix": "classify rte: ",
    },
    "boolq": {
        "hf_name": ("super_glue", "boolq"),
        "text_key": None,
        "label_key": "label",
        "labels": ["false", "true"],
        "prefix": "answer question: ",
    },
    "cb": {
        "hf_name": ("super_glue", "cb"),
        "text_key": None,
        "label_key": "label",
        "labels": ["entailment", "contradiction", "neutral"],
        "prefix": "classify cb: ",
    },
    "copa": {
        "hf_name": ("super_glue", "copa"),
        "text_key": None,
        "label_key": "label",
        "labels": ["choice1", "choice2"],
        "prefix": "classify copa: ",
    },
    "wic": {
        "hf_name": ("super_glue", "wic"),
        "text_key": None,
        "label_key": "label",
        "labels": ["false", "true"],
        "prefix": "classify wic: ",
    },
    "multirc": {
        "hf_name": ("super_glue", "multirc"),
        "text_key": None,
        "label_key": "label",
        "labels": ["false", "true"],
        "prefix": "classify multirc: ",
    },
    "imdb": {
        "hf_name": "imdb",
        "text_key": "text",
        "label_key": "label",
        "labels": ["negative", "positive"],
        "prefix": "classify sentiment: ",
    },
    "sst2_v2": {  # second copy of sst2 with different split to avoid overlap
        "hf_name": ("glue", "sst2"),
        "text_key": "sentence",
        "label_key": "label",
        "labels": ["negative", "positive"],
        "prefix": "classify review: ",
    },
}


class CLTaskDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _format_input(task_name: str, example: dict) -> str:
    cfg = TASK_CONFIG[task_name]
    prefix = cfg["prefix"]

    if task_name == "mnli":
        return prefix + "premise: " + example["premise"] + " hypothesis: " + example["hypothesis"]
    elif task_name in ("qqp",):
        return prefix + "question1: " + example["question1"] + " question2: " + example["question2"]
    elif task_name in ("rte",):
        return prefix + "sentence1: " + example["sentence1"] + " sentence2: " + example["sentence2"]
    elif task_name == "boolq":
        return prefix + "question: " + example["question"] + " passage: " + example["passage"][:200]
    elif task_name in ("cb",):
        return prefix + "premise: " + example["premise"] + " hypothesis: " + example["hypothesis"]
    elif task_name == "copa":
        return prefix + "premise: " + example["premise"] + " choice1: " + example["choice1"] + " choice2: " + example["choice2"]
    elif task_name == "wic":
        return prefix + "word: " + example["word"] + " sentence1: " + example["sentence1"] + " sentence2: " + example["sentence2"]
    elif task_name == "multirc":
        return prefix + "paragraph: " + example["paragraph"][:200] + " question: " + example["question"] + " answer: " + example["answer"]
    else:
        text_key = cfg["text_key"]
        return prefix + str(example[text_key])[:300]


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
    max_input_len: int = 256,
    max_target_len: int = 8,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[List[dict], DataLoader, DataLoader]:
    """
    Load, tokenise, and return (train_list, train_loader, test_loader).

    train_list: raw list of tokenised dicts (for buffer operations).
    """
    cfg = TASK_CONFIG[task_name]
    hf_name = cfg["hf_name"]

    if isinstance(hf_name, tuple):
        raw = load_dataset(*hf_name)
    else:
        raw = load_dataset(hf_name)

    train_split = "train"
    if "test_split" in cfg:
        test_split = cfg["test_split"]
    elif "validation" in raw:
        test_split = "validation"
    else:
        test_split = "test"

    train_raw = raw[train_split].shuffle(seed=seed).select(range(min(n_train, len(raw[train_split]))))
    test_raw = raw[test_split].shuffle(seed=seed).select(range(min(n_test, len(raw[test_split]))))

    label_list = cfg["labels"]

    def tokenise(example):
        input_text = _format_input(task_name, example)
        label_id = example[cfg["label_key"]]
        target_text = label_list[label_id]

        model_inputs = tokenizer(
            input_text,
            max_length=max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = tokenizer(
            text_target=target_text,
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

    train_samples = [tokenise(ex) for ex in train_raw]
    test_samples = [tokenise(ex) for ex in test_raw]

    train_ds = CLTaskDataset(train_samples)
    test_ds = CLTaskDataset(test_samples)

    pin = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=_collate_fn, num_workers=num_workers, pin_memory=pin)

    return train_samples, train_loader, test_loader
