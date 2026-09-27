"""Compatibility entry point for the paper's leakage-safe CORAL procedure.

Use ``scripts/apply_coral.py`` for new runs. This wrapper preserves the former
MUSK command while delegating to the single implementation used by all
encoders.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apply_coral import main


if __name__ == "__main__":
    if "--feature-root" not in sys.argv:
        sys.argv.extend(["--feature-root", "data/features/MUSK"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "data/features/MUSK_CORAL"])
    raise SystemExit(main())
