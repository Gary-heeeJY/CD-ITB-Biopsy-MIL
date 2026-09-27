import torch


class Config:
    # ===== Feature extraction paths =====
    LABEL_FILE = '<LABEL_FILE>'

    PATCH_DIR = './data/patches'
    H5_EXT = '.h5'
    SLIDE_DIR = './data/wsi'
    SLIDE_EXT = '.tif'

    FEATURE_DIR = './data/features/CONCH'

    # ===== CONCH model =====
    CONCH_MODEL_PATH = './weights/CONCH/conch_checkpoint.pt'
    CONCH_MODEL_CFG = 'conch_vitb'

    # Official CONCH recommendation for WSI/MIL feature extraction is to use image embeddings
    # before the contrastive projection head and before normalization.
    # If your old experiment intentionally used retrieval-style 512-D features, set both to True
    # and set EXPECTED_FEATURE_DIM accordingly after confirming the actual output dimension.
    CONCH_PROJ_CONTRAST = False
    CONCH_NORMALIZE = False

    # Set to None to auto-detect feature dimension and write it to metadata.
    # For many CONCH ViT-B configurations, proj_contrast=False may output pre-projection features
    # whose dimension differs from the retrieval-style projected features.
    EXPECTED_FEATURE_DIM = None

    IMG_SIZE = 448
    TEXT_MAX_LEN = 128
    BATCH_SIZE_Extract_Feature = 256
    MAX_NUM_WORKERS = 4
    FEATURE_EXTRACTION_MODE = 1  # 1: skip existing complete outputs; 2: overwrite/re-extract

    # ===== Training-related values, kept for compatibility with your project =====
    BATCH_SIZE = 1
    EPOCHS = 100
    HIDDEN_DIM = None  # Set downstream DATA.hidden_dim to the feature_dim written in metadata.
    VAL_SPLIT = 0.3
    WARMUP_EPOCHS = 20
    MAX_LR = 2e-5
    MIN_LR = 1e-8
    DROPOUT = 0.4
    WEIGHT_DECAY = 0.0003
    BETA1 = 0.9
    BETA2 = 0.999
    AUGMENT = True
    AUG_NOISE_STD = 0.05
    AUG_DROPOUT_RATE = 0.1
    AUG_SCALING_RANGE = [0.9, 1.1]
    SCALES = [1]

    GPU_ID = 0
    DEVICE = f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu'

    OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)
