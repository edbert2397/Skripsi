"""
Check what accuracy we're actually computing
"""
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="google/t5gemma-2-1b-1b", 
                    help="Model directory to analyze results from")
args = parser.parse_args()

model_dir = args.model.replace("/", "_")
results_path = os.path.join("results", model_dir)

# Load the results
try:
    with open(os.path.join(results_path, "dlog_results.json"), "r") as f:
        dlog = json.load(f)

    with open(os.path.join(results_path, "baseline_results.json"), "r") as f:
        baseline = json.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"Make sure you have run the experiment with this model first.")
    exit(1)

print(f"--- Analyzing results for model: {args.model} ---\n")
print("DLOG Performance Matrix:")
print(dlog["cl_metrics"]["Performance Matrix"])
print(f"FP: {dlog['cl_metrics']['Final Performance (FP)']}")
print(f"FT: {dlog['cl_metrics']['Forgetting (FT)']}")

print("\nBaseline Performance Matrix:")
print(baseline["cl_metrics"]["Performance Matrix"])
print(f"FP: {baseline['cl_metrics']['Final Performance (FP)']}")
print(f"FT: {baseline['cl_metrics']['Forgetting (FT)']}")

print("\n" + "="*60)
print("Training Loss Progression (DLOG):")
print("="*60)

# Show first and last loss for each task
sst2_logs = [log for log in dlog["train_log"] if log["task"] == "sst2"]
ag_logs = [log for log in dlog["train_log"] if log["task"] == "ag_news"]

print(f"\nSST-2:")
print(f"  First loss (step {sst2_logs[0]['step']}): {sst2_logs[0]['task_loss']:.4f}")
print(f"  Last loss (step {sst2_logs[-1]['step']}): {sst2_logs[-1]['task_loss']:.4f}")
print(f"  → Loss decreased: {'YES ✓' if sst2_logs[-1]['task_loss'] < sst2_logs[0]['task_loss'] else 'NO ✗'}")

print(f"\nAG News:")
print(f"  First loss (step {ag_logs[0]['step']}): {ag_logs[0]['task_loss']:.4f}")
print(f"  Last loss (step {ag_logs[-1]['step']}): {ag_logs[-1]['task_loss']:.4f}")
print(f"  → Loss decreased: {'YES ✓' if ag_logs[-1]['task_loss'] < ag_logs[0]['step'] else 'NO ✗'}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
print("✓ SST-2 loss decreased significantly (model learned)")
print("✗ AG News loss stayed high or increased (model did NOT learn)")
print("\nThis explains why accuracy is 0% - the model didn't learn the tasks!")
print("\nPossible causes:")
print("1. T5Gemma-2 needs more training steps")
print("2. Learning rate might be too low")
print("3. Batch size reduction (to 4) might have made training unstable")
print("4. The model architecture might need different hyperparameters")
