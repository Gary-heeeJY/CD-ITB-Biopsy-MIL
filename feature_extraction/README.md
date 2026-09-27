# Feature extraction

All extractors use the same local inputs:

```text
data/wsi/<slide_id>.<wsi_extension>
data/patches/<slide_id>.h5
```

The H5 file must contain `coords` and the attributes `patch_level` and
`patch_size`. Each extractor writes one slide directory containing
`image_feature.pt` and metadata files under `data/features/`.

| Extractor | Entry point | Output dimension | Weight source |
| --- | --- | ---: | --- |
| MUSK | `musk/extract_features_musk.py` | 1024 | Official MUSK checkpoint |
| UNI2-h | `uni/extract_features_uni2-h.py` | 1536 | Official UNI2-h checkpoint |
| CONCH | `conch/extract_features_conch.py` | Detected and recorded | Official CONCH checkpoint |
| ImageNet ResNet50 | `resnet50/extract_features_resnet50.py` | 2048 | torchvision ImageNet weights |

All four extractors apply Macenko stain normalization internally, before their
encoder-specific transform. MUSK uses `ms_aug=False`. All encoders are frozen;
only the downstream MIL models are trained in this repository. There is no
second `*_macenko.py` experiment path.

See [../docs/PREPROCESSING.md](../docs/PREPROCESSING.md) for exact patch
transforms and [../docs/MODEL_SOURCES.md](../docs/MODEL_SOURCES.md) for
official code, weights, and citation links.
