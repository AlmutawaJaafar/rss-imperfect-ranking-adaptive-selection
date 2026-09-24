#!/usr/bin/env python3
"""Generate Figures 1-11 with their complete panel sets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geoheavytails.figures import ALL_FIGURES  # noqa: E402


def _jsonable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def generate(numbers=None, verbose=True):
    numbers = sorted(ALL_FIGURES) if numbers is None else sorted(numbers)
    summary = {}
    for n in numbers:
        t0 = time.time()
        result = ALL_FIGURES[n]()
        summary[n] = result
        if verbose:
            print(f"Figure {n:2d}  ({time.time()-t0:5.1f} s)  "
                  f"-> {Path(result['path']).name}")
    out = ROOT / "data" / "figure_summary.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(_jsonable(summary), f, indent=2, default=str)
    if verbose:
        print(f"\nQuantities quoted in captions written to {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("figures", nargs="*", type=int,
                    help="figure numbers (default: all)")
    args = ap.parse_args()
    generate(args.figures or None)
