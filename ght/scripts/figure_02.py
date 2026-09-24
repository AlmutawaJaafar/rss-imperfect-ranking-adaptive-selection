#!/usr/bin/env python3
"""Generate Figure 2 only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geoheavytails.figures import ALL_FIGURES  # noqa: E402

if __name__ == "__main__":
    result = ALL_FIGURES[2]()
    print(f"Generated Figure 2 -> {result['path']}")
