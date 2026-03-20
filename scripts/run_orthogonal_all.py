#!/usr/bin/env python3
"""
run_orthogonal_all.py - Run ALL orthogonal methods (O-Grad + O-Feat + O-Conflict) together.

Runs all three orthogonal methods against baselines in a single comparison.
This generates a complete head-to-head comparison plot with all 5 methods:
O-Grad, O-Feat, O-Conflict, SuRe (Surprise), Reservoir (Random).

Usage:
  # Full comparison (all 4 methods, all benchmarks):
  python scripts/run_orthogonal_all.py

  # Quick sanity check:
  python scripts/run_orthogonal_all.py --fast

  # Specific benchmarks/seeds:
  python scripts/run_orthogonal_all.py --benchmarks lnt --seeds 42 123

  # After completion, generate aggregated comparison plots:
  python scripts/plot_results.py --results-dir outputs/ --comparison
"""

import subprocess
import sys
from pathlib import Path

COMPARISON_SCRIPT = Path(__file__).parent / "run_comparison.py"


def main():
    args = sys.argv[1:]

    cmd = [
        sys.executable, str(COMPARISON_SCRIPT),
        "--methods", "orthogonal", "feature", "conflict", "surprise", "reservoir",
    ] + args

    print("=" * 60)
    print("[ALL] Running Method 1 (O-Grad) + Method 2 (O-Feat) + Method 3 (O-Conflict) + Baselines")
    print("=" * 60)
    print(f">>> {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        # Auto-generate aggregated comparison plots
        plot_script = Path(__file__).parent / "plot_results.py"
        print("\n[Plot] Generating aggregated comparison plots...")
        subprocess.run([
            sys.executable, str(plot_script),
            "--results-dir", "outputs",
            "--comparison",
        ])

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
