import json
import numpy as np
from pathlib import Path

LOG = Path("outputs/rq1_ord0_s42_orthogonal_ema_b0p995/sigma_log_orthogonal.jsonl")
rows = [json.loads(l) for l in open(LOG)]

print(f"{'task':12s} {'σ₁':>8s} {'σ₅':>8s} {'σ₁₀':>8s} {'σ₃₁':>8s} "
      f"{'top5/all':>10s} {'top10/all':>10s} {'eff@95%':>10s} {'eff@99%':>10s}")

ratios_top10 = []
eff_ranks_95 = []

for r in rows:
    sigma = np.array(r["singular_values"])
    energy = sigma ** 2
    if energy.sum() == 0:
        print(f"{r['task_name']:12s}  ALL-ZERO BUG")
        continue
    total = energy.sum()
    cumulative = np.cumsum(energy) / total
    top5 = cumulative[4]
    top10 = cumulative[9]
    eff95 = int(np.searchsorted(cumulative, 0.95) + 1)
    eff99 = int(np.searchsorted(cumulative, 0.99) + 1)
    ratios_top10.append(top10)
    eff_ranks_95.append(eff95)
    print(f"{r['task_name']:12s} {sigma[0]:8.2f} {sigma[4]:8.2f} {sigma[9]:8.2f} {sigma[30]:8.2f} "
          f"{top5:10.1%} {top10:10.1%} {eff95:10d} {eff99:10d}")

print()
print(f"Mean top10/all: {np.mean(ratios_top10):.1%}")
print(f"Mean eff_rank@95%: {np.mean(eff_ranks_95):.1f}")
print(f"Max  eff_rank@95%: {max(eff_ranks_95)}")