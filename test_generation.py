"""
Test evaluation manually to see what's wrong
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data_pipeline import build_task_dataloaders
from config import DLOGConfig

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2-270m-270m")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2-270m-270m")
model = model.to("cuda")
model.eval()

print("Building evaluation dataloader...")
config = DLOGConfig()
_, eval_loaders = build_task_dataloaders(
    tokenizer=tokenizer,
    task_order=["sst2"],
    batch_size=4,
    max_samples_train=50,
    max_samples_eval=20,
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
    
    print(f"\nBatch {i+1}:")
    for j, (pred, label) in enumerate(zip(preds, label_texts)):
        match = "✓" if pred.strip().lower() == label.strip().lower() else "✗"
        print(f"  [{match}] Predicted: '{pred}' | Expected: '{label}'")
        
        if pred.strip().lower() == label.strip().lower():
            correct += 1
        total += 1

print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}")
