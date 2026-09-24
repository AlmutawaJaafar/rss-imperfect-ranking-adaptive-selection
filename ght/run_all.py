#!/usr/bin/env python3
"""
One-command reproduction: figures, validation tables, and the automated
manuscript-consistency check.

    python run_all.py

Exit status is nonzero if any consistency check fails, so this can be wired
into CI and made a precondition for submission.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    name = Path(script).name
    print(f"\n=== {name} " + "=" * max(4, 60 - len(name)))
    return subprocess.run([sys.executable, str(script), *args],
                          cwd=ROOT).returncode


if __name__ == "__main__":
    rc = run(ROOT / "scripts" / "generate_all_figures.py")
    rc |= run(ROOT / "scripts" / "generate_validation_tables.py")
    check = run(ROOT / "scripts" / "check_manuscript_consistency.py")
    print("\n" + "=" * 70)
    if check == 0 and rc == 0:
        print("All figures and tables regenerated; all consistency checks PASS.")
    else:
        print("FAILURES PRESENT -- do not submit until resolved.")
    sys.exit(check or rc)
