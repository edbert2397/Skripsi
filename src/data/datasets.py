"""
datasets.py - Dataset loading and T5 tokenisation for all benchmark tasks.

All tasks are framed as seq2seq: input is "task: text", output is label string.
This matches the SuRe paper's T5-Large setup.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from datasets import load_dataset
from typing import List, Tuple, Dict, Optional


# ---- Task definitions ----
TASK_CONFIG = {
    "ag_news": {
        "hf_name": "fancyzhx/ag_news",
        "text_key": "text",
        "label_key": "label",
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "prefix": "classify news: ",
    },
    "amazon_reviews": {
        "hf_name": "fancyzhx/amazon_polarity",
        "text_key": "content",
        "label_key": "label",
        "labels": ["negative", "positive"],
        "prefix": "classify sentiment: ",
    },
    "dbpedia": {
        "hf_name": "fancyzhx/dbpedia_14",
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
        "hf_name": "community-datasets/yahoo_answers_topics",
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
        "hf_name": ("nyu-mll/glue", "sst2"),
        "text_key": "sentence",
        "label_key": "label",
        "labels": ["negative", "positive"],
        "prefix": "classify sentiment: ",
    },
    "mnli": {
        "hf_name": ("nyu-mll/glue", "mnli"),
        "text_key": None,   # special handling
        "label_key": "label",
        "labels": ["entailment", "neutral", "contradiction"],
        "prefix": "classify nli: ",
        "test_split": "validation_matched",
    },
    "qqp": {
        "hf_name": ("nyu-mll/glue", "qqp"),
        "text_key": None,
        "label_key": "label",
        "labels": ["not duplicate", "duplicate"],
        "prefix": "classify paraphrase: ",
    },
    "rte": {
        "hf_name": ("nyu-mll/glue", "rte"),
        "text_key": None,
        "label_key": "label",
        "labels": ["entailment", "not entailment"],
        "prefix": "classify rte: ",
    },
    "boolq": {
        "hf_name": ("aps/super_glue", "boolq"),
        "text_key": None,
        "label_key": "label",
        "labels": ["false", "true"],
        "prefix": "answer question: ",
    },
    "cb": {
        "hf_name": ("aps/super_glue", "cb"),
        "text_key": None,
        "label_key": "label",
        "labels": ["entailment", "contradiction", "neutral"],
        "prefix": "classify cb: ",
    },
    "copa": {
        "hf_name": ("aps/super_glue", "copa"),
        "text_key": None,
        "label_key": "label",
        "labels": ["choice1", "choice2"],
        "prefix": "classify copa: ",
    },
    "wic": {
        "hf_name": ("aps/super_glue", "wic"),
        "text_key": None,
        "label_key": "label",
        "labels": ["false", "true"],
        "prefix": "classify wic: ",
    },
    "multirc": {
        "hf_name": ("aps/super_glue", "multirc"),
        "text_key": None,
        "label_key": "label",
        "labels": ["false", "true"],
        "prefix": "classify multirc: ",
    },
    "imdb": {
        "hf_name": "stanfordnlp/imdb",
        "text_key": "text",
        "label_key": "label",
        "labels": ["negative", "positive"],
        "prefix": "classify sentiment: ",
    },
    "trec": {
        "hf_name": "SetFit/TREC-QC",
        "text_key": "text",
        "label_key": "label_coarse",
        "labels": ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"],
        "prefix": "classify question: ",
    },
    "snli": {
        "hf_name": "stanfordnlp/snli",
        "text_key": None,   # special: premise + hypothesis
        "label_key": "label",
        "labels": ["entailment", "neutral", "contradiction"],
        "prefix": "classify nli: ",
        "filter_invalid_labels": True,  # SNLI has label=-1 for no-consensus examples
    },
    "cola": {
        "hf_name": ("nyu-mll/glue", "cola"),
        "text_key": "sentence",
        "label_key": "label",
        "labels": ["unacceptable", "acceptable"],
        "prefix": "classify acceptability: ",
    },
    "yelp": {
        "hf_name": "fancyzhx/yelp_polarity",
        "text_key": "text",
        "label_key": "label",
        "labels": ["negative", "positive"],
        "prefix": "classify sentiment: ",
    },
    "qnli": {
        "hf_name": ("nyu-mll/glue", "qnli"),
        "text_key": None,   # special: question + sentence
        "label_key": "label",
        "labels": ["entailment", "not_entailment"],
        "prefix": "classify qnli: ",
    },
    "mrpc": {
        "hf_name": ("nyu-mll/glue", "mrpc"),
        "text_key": None,   # special: sentence1 + sentence2
        "label_key": "label",
        "labels": ["not_equivalent", "equivalent"],
        "prefix": "classify paraphrase: ",
    },
    "20news": {
        "hf_name": "SetFit/20_newsgroups",
        "text_key": "text",
        "label_key": "label",
        "labels": [
            "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
            "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
            "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
            "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
            "sci.space", "soc.religion.christian", "talk.politics.guns",
            "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"
        ],
        "prefix": "classify newsgroup: ",
    },
}


def _balanced_sample(hf_dataset, label_key: str, n_total: int, seed: int):
    """
    Return a balanced HF dataset subset: exactly (n_total // n_classes) examples
    per class, shuffled with the given seed.

    - Rows with label < 0 are silently dropped before sampling (handles SNLI -1).
    - If a class has fewer rows than n_per_class, all available rows are taken.
    - Using numpy default_rng(seed) (not global state) guarantees that two
      independent runs with the same seed pick identical row indices.
    """
    rng = np.random.default_rng(seed)

    all_labels = hf_dataset[label_key]
    label_to_indices: Dict[int, list] = {}
    for i, lbl in enumerate(all_labels):
        if lbl < 0:
            continue
        label_to_indices.setdefault(lbl, []).append(i)

    unique_labels = sorted(label_to_indices.keys())
    n_classes = len(unique_labels)
    n_per_class = n_total // n_classes

    selected: List[int] = []
    for lbl in unique_labels:
        idxs = np.array(label_to_indices[lbl])
        perm = rng.permutation(len(idxs))
        chosen = idxs[perm[: min(n_per_class, len(idxs))]]
        selected.extend(chosen.tolist())

    selected = rng.permutation(selected).tolist()
    return hf_dataset.select(selected)


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

    if task_name in ("mnli", "snli"):
        return prefix + "premise: " + example["premise"] + " hypothesis: " + example["hypothesis"]
    elif task_name in ("qqp",):
        return prefix + "question1: " + example["question1"] + " question2: " + example["question2"]
    elif task_name in ("rte", "mrpc"):
        return prefix + "sentence1: " + example["sentence1"] + " sentence2: " + example["sentence2"]
    elif task_name == "qnli":
        return prefix + "question: " + example["question"] + " sentence: " + example["sentence"]
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
    balanced: bool = True,
) -> Tuple[List[dict], DataLoader, DataLoader]:
    """
    Load, tokenise, and return (train_list, train_loader, test_loader).

    train_list: raw list of tokenised dicts (for buffer operations).

    When balanced=True (default), each class contributes exactly
    n_train // n_classes (and n_test // n_classes) examples, selected
    with numpy.random.default_rng(seed) so results are fully reproducible
    across machines given the same seed.
    """
    cfg = TASK_CONFIG[task_name]
    hf_name = cfg["hf_name"]
    label_key = cfg["label_key"]

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

    if balanced:
        train_raw = _balanced_sample(raw[train_split], label_key, n_train, seed)
        test_raw = _balanced_sample(raw[test_split], label_key, n_test, seed)
    else:
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
