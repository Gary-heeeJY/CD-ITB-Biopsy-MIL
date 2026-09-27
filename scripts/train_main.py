"""Train the proposed hierarchical MambaMIL model."""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from cd_itb import run_legacy
from _common import current_args, with_default


if __name__ == "__main__":
    args = with_default(current_args(), "--cfg", "configs/MambaMIL_DynamicWeight.yaml")
    run_legacy("scripts/train_MambaMIL.py", args)
