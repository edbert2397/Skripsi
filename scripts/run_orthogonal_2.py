#!/usr/bin/env python3
"""
run_orthogonal_2.py - Run O-Feat (Method 2: Feature Subspace Orthogonality) only.

Runs the feature-based orthogonal selection method against baselines on RQ1.
This script does NOT run O-Grad, so it won't overwrite O-Grad results.

Usage:
  # Full comparison (O-Feat vs SuRe vs Reservoir):
  python scripts/run_orthogonal_2.py

  # Quick sanity check:
  python scripts/run_orthogonal_2.py --fast

  # Specific seeds:
  python scripts/run_orthogonal_2.py --seeds 42
"""

import subprocess
import sys
from pathlib import Path

COMPARISON_SCRIPT = Path(__file__).parent / "run_comparison.py"


def main():
    # Forward all CLI args, but force --methods to feature + baselines
    args = sys.argv[1:]

    cmd = [
        sys.executable, str(COMPARISON_SCRIPT),
        "--methods", "feature", "surprise", "reservoir",
        "--benchmarks", "rq1",
        "--orders", "0",
    ] + args

    print("=" * 60)
    print("[O-Feat] Running Method 2: Feature Subspace Orthogonality")
    print("=" * 60)
    print(f">>> {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
