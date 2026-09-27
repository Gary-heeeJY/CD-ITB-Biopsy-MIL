from .dataset_generic import build_dataset
from .utils import make_datalist

def get_dataset(cfg, datalist):
    dataset = {
        "biospy_CD_ITB": build_dataset(cfg, datalist),
        # "CAMELYON17": build_dataset_camelyon17(cfg, datalist)
    }
    return dataset.get(cfg.DATA.type, None)

def build_data_loader(cfg):
    datalist = make_datalist(cfg)
    loaders = get_dataset(cfg, datalist)

    return loaders