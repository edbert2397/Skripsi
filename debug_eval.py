"""
Quick debug script to see what the model is actually generating
"""
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from config import DATASET_CONFIGS

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2-270m-270m")

# Load a few samples from ag_news
ds = load_dataset("ag_news", split="test[:10]")

cfg = DATASET_CONFIGS["ag_news"]
class_names = cfg["class_names"]

print("AG News Evaluation Debug")
print("=" * 60)
print(f"Class names: {class_names}\n")

for i, item in enumerate(ds):
    input_text = cfg["prompt_template"].format(text=item["text"][:100])
    label_idx = item["label"]
    label_text = class_names[label_idx]
    
    print(f"\nSample {i+1}:")
    print(f"Input: {input_text}")
    print(f"Expected Label: {label_text} (idx={label_idx})")
    
    # Show what the tokenizer encodes
    inputs = tokenizer(label_text, return_tensors="pt")
    decoded = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
    print(f"Tokenized label decodes to: '{decoded}'")
    
    if i >= 2:  # Just show first 3
        break

print("\n" + "=" * 60)
print("Now checking SST-2...")
print("=" * 60)

ds_sst = load_dataset("glue", "sst2", split="validation[:5]")
cfg_sst = DATASET_CONFIGS["sst2"]
class_names_sst = cfg_sst["class_names"]

print(f"Class names: {class_names_sst}\n")

for i, item in enumerate(ds_sst):
    input_text = cfg_sst["prompt_template"].format(text=item["sentence"][:100])
    label_idx = item["label"]
    label_text = class_names_sst[label_idx]
    
    print(f"\nSample {i+1}:")
    print(f"Input: {input_text}")
    print(f"Expected Label: {label_text} (idx={label_idx})")
    
    # Show what the tokenizer encodes
    inputs = tokenizer(label_text, return_tensors="pt")
    decoded = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
    print(f"Tokenized label decodes to: '{decoded}'")
    
    if i >= 2:
        break
