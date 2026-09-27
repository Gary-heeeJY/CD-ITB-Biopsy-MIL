import torch


class Config:
    """Relative demo paths and parameters for MUSK feature extraction."""

    LABEL_FILE = "./data/labels/labels.csv"
    PATCH_DIR = "./data/patches"
    H5_EXT = ".h5"
    SLIDE_DIR = "./data/wsi"
    SLIDE_EXT = ".tif"
    SLIDE_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")
    FEATURE_DIR = "./data/features/MUSK"
    VAL_FEATURE_DIR = "./outputs/features"
    MODEL_SAVE_DIR = "./outputs/feature_extraction"

    MODEL_NAME = "musk_large_patch16_384"
    MODEL_KEY = "model|module"
    TOKENIZER_LOCAL_PATH = "./weights/MUSK/tokenizer"
    MUSK_MODEL_LOCAL_PATH = "./weights/MUSK/musk.pth"

    IMG_SIZE = 384
    TEXT_MAX_LEN = 1000
    BATCH_SIZE_Extract_Feature = 512
    CUSTOM_DOWNSAMPLE = 1
    # CLAM-style patches are 512 x 512 pixels at the maximum-resolution level.
    TARGET_PATCH_SIZE = 512
    BATCH_SIZE = 128
    MAX_NUM_WORKERS = 4
    EPOCHS = 200
    HIDDEN_DIM = 512
    VAL_SPLIT = 0.3
    WARMUP_EPOCHS = 100
    MAX_LR = 5e-5
    MIN_LR = 1e-8
    DROPOUT = 0.1
    WEIGHT_DECAY = 0.05
    BETA1 = 0.9
    BETA2 = 0.999
    AUGMENT = False
    AUG_NOISE_STD = 0.1
    AUG_DROPOUT_RATE = 0.2
    AUG_SCALING_RANGE = [0.9, 1.1]
    SCALES = [1, 2]

    GPU_ID = 0
    DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"


config = Config()
