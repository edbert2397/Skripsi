"""evaluator.py - Metrics: Average Performance (AP), Final Performance (FP), Backward Transfer (BWT)."""

import torch
from typing import Dict, List
from transformers import AutoTokenizer


class Evaluator:
    def __init__(self, device: torch.device, use_fp16: bool = True):
        self.device = device
        self.use_fp16 = use_fp16
        self.amp_dtype = torch.bfloat16 if (self.use_fp16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        # all_results[task_id_trained][task_id_eval] = accuracy
        self.all_results: Dict[int, Dict[int, float]] = {}
        self.tokenizer = AutoTokenizer.from_pretrained("t5-large", legacy=False, local_files_only=True)

    @torch.no_grad()
    def evaluate_task(self, model, task_id: int, dataloader) -> float:
        """Evaluate slow model accuracy on a single task via generate().

        Uses greedy decoding (max_new_tokens=5) and exact string match against
        the ground-truth label. Requires the slow LoRA to have converged enough
        via EMA for meaningful results.
        """
        model.eval()
        correct = total = 0
        tokenizer = self.tokenizer

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

    def compute_average_performance(self) -> float:
        """AP (Learning Accuracy): mean of diagonal entries R_{j,j}.

        R_{j,j} is the accuracy on task j evaluated immediately after training
        on task j — a pure plasticity measure unaffected by later forgetting.
        Formula: AP = (1/T) * sum_{j=1}^{T} R_{j,j}
        """
        tasks = sorted(self.all_results.keys())
        diagonal = [self.all_results[j][j] for j in tasks if j in self.all_results[j]]
        return sum(diagonal) / len(diagonal) if diagonal else 0.0

    def compute_final_performance(self) -> float:
        """FP (Average Accuracy): mean accuracy across all tasks after the last task.

        Formula: FP = (1/T) * sum_{j=1}^{T} R_{T,j}
        """
        last_task = max(self.all_results.keys())
        return sum(self.all_results[last_task].values()) / len(self.all_results[last_task])

    def compute_backward_transfer(self) -> float:
        """BWT (Backward Transfer): measures how much learning new tasks affects old ones.

        Formula: BWT = (1/(T-1)) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
        - BWT < 0: Catastrophic Forgetting (old tasks overwritten)
        - BWT = 0: No forgetting
        - BWT > 0: Positive Transfer (new learning improved old tasks)
        """
        tasks = sorted(self.all_results.keys())
        T = tasks[-1]
        if len(tasks) < 2:
            return 0.0
        bwt_scores = [
            self.all_results[T][i] - self.all_results[i][i]
            for i in tasks[:-1]  # i = 1 to T-1
            if i in self.all_results[T] and i in self.all_results[i]
        ]
        return sum(bwt_scores) / len(bwt_scores) if bwt_scores else 0.0

    def compute_forgetting(self) -> float:
        """Forgetting = -BWT: how much old-task accuracy dropped.

        Forgetting = (1/(T-1)) * sum_{i=1}^{T-1} (R_{i,i} - R_{T,i})
        - Forgetting > 0: old tasks degraded (catastrophic forgetting)
        - Forgetting = 0: no forgetting
        - Forgetting < 0: positive backward transfer
        Lower is better.
        """
        return -self.compute_backward_transfer()

    def summary(self) -> Dict[str, float]:
        ap = self.compute_average_performance()
        fp = self.compute_final_performance()
        forgetting = self.compute_forgetting()
        return {"AP": ap, "FP": fp, "Forgetting": forgetting}
