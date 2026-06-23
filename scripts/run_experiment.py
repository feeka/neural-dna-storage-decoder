"""Thin launcher for the full DNA-decoder experiment.

Run with the package installed in editable mode::

    pip install -e .
    python scripts/run_experiment.py --quick
"""
from __future__ import annotations

import argparse

from dnadecoder.experiment import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the dnadecoder experiment.")
    parser.add_argument("--quick", action="store_true",
                        help="tiny sub-minute smoke run")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="directory for results.md and ser_vs_noise.png")
    args = parser.parse_args()

    out = run_experiment(quick=args.quick, out_dir=args.out_dir)
    print(f"done. artifacts written to: {out['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
