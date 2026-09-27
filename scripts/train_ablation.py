"""Train one of the reported ablation settings."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cd_itb import run_legacy
from _common import current_args, has_option


VARIANTS = {
    "meanpool": ("scripts/train_MambaMIL_A_meanpool.py", "configs/MambaMIL_DynamicWeight_A_meanpool.yaml"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    known, rest = parser.parse_known_args(current_args())
    script, default_cfg = VARIANTS[known.variant]
    if not has_option(rest, "--cfg"):
        rest.extend(["--cfg", default_cfg])
    run_legacy(script, rest)
       rest.extend(["--cfg", default_cfg])
    run_legacy(script, rest)
