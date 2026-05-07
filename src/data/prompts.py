"""
prompts.py - Per-task prompt formatting for CL14 inputs.

Single-text tasks append an `Options: ...` line after the text.
Pair tasks split the CSV `text` column on `[SEP]` (case-insensitive)
and slot each half into a Premise/Hypothesis-style template.
"""

import re
from typing import List

OPTIONS_BY_TASK = {
    "sst2":        ["positive", "negative"],
    "ag_news":     ["World", "Sports", "Business", "Sci/Tech"],
    "imdb":        ["pos", "neg"],
    "cola":        ["acceptable", "unacceptable"],
    "yelp":        ["positive", "negative"],
    "yahoo": [
        "Society & Culture", "Science & Mathematics", "Health",
        "Education & Reference", "Computers & Internet", "Sports",
        "Business & Finance", "Entertainment & Music",
        "Family & Relationships", "Politics & Government",
    ],
    "newsgroup20": [
        "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x",
        "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball",
        "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
        "sci.space", "soc.religion.christian", "talk.politics.guns",
        "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc",
    ],
    "mnli":     ["entailment", "neutral", "contradiction"],
    "snli":     ["entailment", "neutral", "contradiction"],
    "rte":      ["entailment", "not_entailment"],
    "qnli":     ["entailment", "not_entailment"],
    "mrpc":     ["equivalent", "not_equivalent"],
    "dbpedia": [
        "Company", "EducationalInstitution", "Artist", "Athlete",
        "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
        "Village", "Animal", "Plant", "Album", "Film", "WrittenWork",
    ],
    "amazon":   ["positive", "negative"],
}

# Pair tasks have two segments joined by [SEP] in the CSV `text` column.
# Each entry maps to (left_label, right_label) for the prompt template.
PAIR_TEMPLATES = {
    "mnli": ("Premise",   "Hypothesis"),
    "snli": ("Premise",   "Hypothesis"),
    "rte":  ("Premise",   "Hypothesis"),
    "qnli": ("Question",  "Sentence"),
    "mrpc": ("Sentence 1", "Sentence 2"),
}

_SEP_RE = re.compile(r"\s*\[sep\]\s*", flags=re.IGNORECASE)


def _options_line(task_name: str) -> str:
    return "Options: " + ", ".join(OPTIONS_BY_TASK[task_name])


def format_prompt(task_name: str, text: str) -> str:
    """Render a task-specific prompt for T5 input."""
    if task_name not in OPTIONS_BY_TASK:
        raise KeyError(f"Unknown task '{task_name}' — no prompt template defined.")

    if task_name in PAIR_TEMPLATES:
        left_label, right_label = PAIR_TEMPLATES[task_name]
        parts = _SEP_RE.split(text, maxsplit=1)
        # Defensive: if no [SEP] found, treat the whole text as the left part.
        left = parts[0].strip()
        right = parts[1].strip() if len(parts) == 2 else ""
        return (
            f"{left_label}: {left}\n"
            f"{right_label}: {right}\n"
            f"{_options_line(task_name)}"
        )

    return f"{text}\n{_options_line(task_name)}"
