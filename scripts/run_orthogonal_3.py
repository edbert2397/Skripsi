#!/usr/bin/env python3
"""
run_orthogonal_3.py - Run O-Conflict (Method 3: Gradient Conflict Score) only.

Runs the gradient conflict-based selection method against baselines.
This script does NOT run O-Grad or O-Feat, so it won't overwrite their results.

Usage:
  # Full comparison (O-Conflict vs SuRe vs Reservoir):
  python scripts/run_orthogonal_3.py

  # Quick sanity check:
  python scripts/run_orthogonal_3.py --fast

  # Specific benchmarks/seeds:
  python scripts/run_orthogonal_3.py --benchmarks standard_cl --seeds 42
"""

import subprocess
import sys
from pathlib import Path

COMPARISON_SCRIPT = Path(__file__).parent / "run_comparison.py"


def main():
    # Forward all CLI args, but force --methods to conflict + baselines
    args = sys.argv[1:]

    cmd = [
        sys.executable, str(COMPARISON_SCRIPT),
        "--methods", "conflict", "surprise", "reservoir",
    ] + args

    print("=" * 60)
    print("[O-Conflict] Running Method 3: Gradient Conflict Score")
    print("=" * 60)
    print(f">>> {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
