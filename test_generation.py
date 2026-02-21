"""
Test evaluation manually to see what's wrong
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data_pipeline import build_task_dataloaders
from config import DLOGConfig

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2-1b-1b")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2-1b-1b")
model = model.to("cuda")
model.eval()

print("Building evaluation dataloader...")
config = DLOGConfig()
config.task_order = ["sst2"]
config.batch_size = 4
train_loaders, eval_loaders = build_task_dataloaders(
    config=config,
    tokenizer=tokenizer,
    max_train_samples=50,
    max_eval_samples=20,
)

print("\nTesting generation on SST-2...")
eval_loader = eval_loaders["sst2"]

correct = 0
total = 0

for i, batch in enumerate(eval_loader):
    if i >= 2:  # Just test 2 batches
        break
    
    input_ids = batch["input_ids"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")
    label_texts = batch["label_text"]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False,
        )
    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"DEBUG: outputs.shape = {outputs.shape}")
    
    print(f"\nBatch {i+1}:")
    for j, (pred, label) in enumerate(zip(preds, label_texts)):
        match = "✓" if pred.strip().lower() == label.strip().lower() else "✗"
        print(f"  [{match}] Predicted: '{pred}' | Expected: '{label}'")
        
        if pred.strip().lower() == label.strip().lower():
            correct += 1
        total += 1

print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}")
