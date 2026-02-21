import sys
import traceback

try:
    from config import DLOGConfig
    from run_experiment import run_experiment
    config = DLOGConfig(model_name='Qwen/Qwen2.5-1.5B')
    run_experiment(config, smoke_test=True)
except Exception as e:
    with open('err.txt', 'w') as f:
        traceback.print_exc(file=f)
    print("Caught Exception:", e)
    sys.exit(1)
