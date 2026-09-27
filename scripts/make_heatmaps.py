"""Generate attention heatmaps from WSI files, coordinates, features, and checkpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cd_itb import run_legacy
from _common import current_args


if __name__ == "__main__":
    run_legacy("Heatmap/create_heatmaps.py", current_args())
