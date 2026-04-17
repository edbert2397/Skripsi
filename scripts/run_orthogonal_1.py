#!/usr/bin/env python3
"""
run_orthogonal_1.py - Run O-Grad (Method 1: Gradient Subspace Orthogonality) only.

Runs the orthogonal gradient-based selection method against baselines on RQ1.
This script does NOT run O-Feat, so it won't overwrite O-Feat results.

Usage:
  # Full comparison (O-Grad vs SuRe vs Reservoir):
  python scripts/run_orthogonal_1.py

  # Quick sanity check:
  python scripts/run_orthogonal_1.py --fast

  # Specific seeds:
  python scripts/run_orthogonal_1.py --seeds 42
"""

import subprocess
import sys
from pathlib import Path

COMPARISON_SCRIPT = Path(__file__).parent / "run_comparison.py"


def main():
    # Forward all CLI args, but force --methods to orthogonal + baselines
    args = sys.argv[1:]

    cmd = [
        sys.executable, str(COMPARISON_SCRIPT),
        "--methods", "orthogonal", "surprise", "reservoir",
        "--benchmarks", "rq1",
        "--orders", "0",
    ] + args

    print("=" * 60)
    print("[O-Grad] Running Method 1: Gradient Subspace Orthogonality")
    print("=" * 60)
    print(f">>> {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
