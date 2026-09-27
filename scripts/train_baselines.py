"""Train the direct patch-to-patient MIL baselines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cd_itb import run_legacy
from _common import current_args, with_default


if __name__ == "__main__":
    args = current_args()
    args = with_default(args, "--abmil-cfg", "configs/biospy_case_abmil.yaml")
    args = with_default(args, "--transmil-cfg", "configs/TransMIL.yaml")
    args = with_default(args, "--mamba-cfg", "configs/MambaMIL.yaml")
    run_legacy("scripts/train_patient_mil_baselines.py", args)
