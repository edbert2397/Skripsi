
import torch
from config import DLOGConfig
from dlog_model import BaselineModel
import gc

def test_baseline_load():
    config = DLOGConfig(model_name="Qwen/Qwen2.5-1.5B", device="cuda")
    print("Attempting to load BaselineModel...")
    try:
        model = BaselineModel(config)
        print("Successfully loaded BaselineModel!")
        print("Model device:", next(model.parameters()).device)
    except Exception as e:
        print("Failed to load BaselineModel:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_baseline_load()
