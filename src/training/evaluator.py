"""evaluator.py - Metrics: Final Performance, Average Performance, Forgetting."""

import torch
from typing import Dict, List


class Evaluator:
    def __init__(self, device: torch.device, use_fp16: bool = True):
        self.device = device
        self.use_fp16 = use_fp16
        self.amp_dtype = torch.bfloat16 if (self.use_fp16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        # all_results[task_id_trained][task_id_eval] = accuracy
        self.all_results: Dict[int, Dict[int, float]] = {}

    @torch.no_grad()
    def evaluate_task(self, model, task_id: int, dataloader) -> float:
        """Evaluate slow model accuracy on a single task via generate().

        Uses greedy decoding (max_new_tokens=5) and exact string match against
        the ground-truth label. Requires the slow LoRA to have converged enough
        via EMA for meaningful results.
        """
        model.eval()
        correct = total = 0

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-large", legacy=False)

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=5,
                        do_sample=False,
                    )
            else:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                    do_sample=False,
                )

            preds_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            labels_obj = labels.clone()
            labels_obj[labels_obj == -100] = tokenizer.pad_token_id
            labels_str = tokenizer.batch_decode(labels_obj, skip_special_tokens=True)

            for p, l in zip(preds_str, labels_str):
                if p.strip().lower() == l.strip().lower():
                    correct += 1
                total += 1

        model.train()
        return correct / total if total > 0 else 0.0

    def evaluate_all_tasks(
        self,
        model,
        task_dataloaders: Dict[int, object],
        after_task_id: int,
    ) -> Dict[int, float]:
        results = {}
        for tid, loader in task_dataloaders.items():
            if tid <= after_task_id:
                results[tid] = self.evaluate_task(model, tid, loader)
        self.all_results[after_task_id] = results
        return results

    def compute_final_performance(self) -> float:
        """FP: average accuracy across all tasks after the last task."""
        last_task = max(self.all_results.keys())
        return sum(self.all_results[last_task].values()) / len(self.all_results[last_task])

    def compute_average_performance(self) -> float:
        """AP: average accuracy across all tasks and all evaluation points."""
        scores = []
        for task_results in self.all_results.values():
            scores.extend(task_results.values())
        return sum(scores) / len(scores) if scores else 0.0

    def compute_forgetting(self) -> float:
        """F = AP - FP (lower is better; negative = improvement over time)."""
        return self.compute_average_performance() - self.compute_final_performance()

    def summary(self) -> Dict[str, float]:
        fp = self.compute_final_performance()
        ap = self.compute_average_performance()
        f = self.compute_forgetting()
        return {"FP": fp, "AP": ap, "Forgetting": f}
