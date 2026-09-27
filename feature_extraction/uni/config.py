import torch


class Config:
    LABEL_FILE = '<LABEL_FILE>'

    PATCH_DIR = './data/patches'
    H5_EXT = '.h5'

    # Both variables are kept for compatibility. The extractor will use SLIDE_DIRS if present.
    SLIDE_DIR = './data/wsi'
    SLIDE_DIRS = [SLIDE_DIR]
    SUPPORTED_SLIDE_EXTS = ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs']

    FEATURE_DIR = './data/features/UNI2-h'
    UNI2_WEIGHT_PATH = './weights/UNI2-h/pytorch_model.bin'

    # Optional local timm path, e.g. '/path/to/third_party/timm_new'
    LOCAL_TIMM_PATH = None

    FEATURE_EXTRACTION_MODE = 1  # 1: skip existing complete outputs; 2: overwrite/re-extract
    BATCH_SIZE_EXTRACT_FEATURE = 256
    MAX_NUM_WORKERS = 4
    MAX_TILES_PER_SLIDE = None
    SEED = 421

    # UNI2-h official architecture uses embed_dim=1536.
    EXPECTED_FEATURE_DIM = 1536

    GPU_ID = 0
    DEVICE = f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu'
