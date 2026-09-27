"""Evaluate the proposed hierarchical MambaMIL model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cd_itb import run_legacy
from _common import current_args, with_default


if __name__ == "__main__":
    args = current_args()
    args = with_default(args, "--cfg", "configs/MambaMIL_DynamicWeight.yaml")
    run_legacy("scripts/test_MambaMIL.py", args)
