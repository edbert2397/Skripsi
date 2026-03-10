"""logging.py - Simple structured logging with optional wandb support."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class Logger:
    def __init__(self, config, run_name: str = None, use_wandb: bool = False):
        self.use_wandb = use_wandb
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{timestamp}"
        self.log_dir = Path("outputs") / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.log_dir / "log.jsonl"

        # Save config
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(vars(config) if hasattr(config, "__dict__") else config, f, indent=2)

        if use_wandb:
            try:
                import wandb
                wandb.init(project="orthogonal-replay", name=self.run_name, config=config)
                self._wandb = wandb
            except ImportError:
                print("[Logger] wandb not installed, skipping.")
                self.use_wandb = False

    def log(self, metrics: Dict[str, Any], step: int = None):
        record = {"step": step, **metrics}
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.use_wandb:
            self._wandb.log(metrics, step=step)

    def log_task_result(self, task_id: int, results: Dict[str, float]):
        self.log({f"task_{k}_acc": v for k, v in results.items()}, step=task_id)
        print(f"[Task {task_id}] {results}")

    def finish(self):
        if self.use_wandb:
            self._wandb.finish()
