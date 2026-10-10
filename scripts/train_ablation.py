"""Train one of the reported ablation settings."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cd_itb import run_legacy
from _common import current_args, has_option


VARIANTS = {
    "meanpool": ("scripts/train_MambaMIL_A_meanpool.py", "configs/MambaMIL_DynamicWeight_A_meanpool.yaml"),
    "attention_only": ("scripts/train_MambaMIL.py", "configs/MambaMIL_DynamicWeight_B_attention_only.yaml", "attention_only"),
    "transformer_only": ("scripts/train_MambaMIL.py", "configs/MambaMIL_DynamicWeight_C_transformer_only.yaml", "transformer_only"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    known, rest = parser.parse_known_args(current_args())
    variant_config = VARIANTS[known.variant]
    script, default_cfg = variant_config[:2]
    if not has_option(rest, "--cfg"):
        rest.extend(["--cfg", default_cfg])
    if len(variant_config) == 3 and not has_option(rest, "--aggregation_variant"):
        rest.extend(["--aggregation_variant", variant_config[2]])
    run_legacy(script, rest)
