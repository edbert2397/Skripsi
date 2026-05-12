import json
import numpy as np
from pathlib import Path

LOG = Path("outputs/rq1_ord0_s42_feature_ema_b0p995/sigma_log_feature.jsonl")
rows = [json.loads(l) for l in open(LOG)]

print(f"{'task':12s} {'σ₁':>8s} {'σ₁₀':>8s} {'σ₁₀₀':>8s} "
      f"{'top10/top100':>14s} {'top20/top100':>14s} "
      f"{'eff_rank@95%':>14s} {'eff_rank@99%':>14s}")

ratios_top10 = []
eff_ranks_95 = []

for r in rows:
    sigma = np.array(r["singular_values"])
    energy = sigma ** 2

    if energy.sum() == 0:
        print(f"{r['task_name']:12s}  ALL-ZERO BUG — skipping")
        continue

    total = energy.sum()
    cumulative = np.cumsum(energy) / total

    top10 = cumulative[9]
    top20 = cumulative[19]
    eff95 = int(np.searchsorted(cumulative, 0.95) + 1)
    eff99 = int(np.searchsorted(cumulative, 0.99) + 1)

    ratios_top10.append(top10)
    eff_ranks_95.append(eff95)

    print(f"{r['task_name']:12s} {sigma[0]:8.1f} {sigma[9]:8.1f} {sigma[99]:8.1f} "
          f"{top10:14.1%} {top20:14.1%} {eff95:14d} {eff99:14d}")

print()
print(f"Mean top10/top100: {np.mean(ratios_top10):.1%}")
print(f"Mean eff_rank@95%: {np.mean(eff_ranks_95):.1f}")
print(f"Max  eff_rank@95%: {max(eff_ranks_95)}")