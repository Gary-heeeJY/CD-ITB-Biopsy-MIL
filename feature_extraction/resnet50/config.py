from __future__ import annotations

from pathlib import Path


class Config:
    """Relative demo paths and defaults for the public ResNet50 extractor."""

    WSI_DIR = Path("./data/wsi")
    PATCH_DIR = Path("./data/patches")
    FEATURE_DIR = Path("./data/features/ResNet50")
    WEIGHTS_PATH = None
    SLIDE_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")
    H5_EXT = ".h5"
    BATCH_SIZE = 128
    NUM_WORKERS = 0
    DEVICE = "cuda"
    SEED = 421
