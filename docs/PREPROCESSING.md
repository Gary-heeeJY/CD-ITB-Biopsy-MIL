# WSI Preprocessing and Feature Specifications

This repository distributes the downstream MIL experiments and reference
feature-extraction code. Patient data, WSIs, H5 coordinate files, and model
weights must be supplied locally. The reported cohort contained 945 WSIs from
233 patients. The internal cohort was used for five-fold patient-level
training/validation and an independent internal test set; the external cohort
was kept as a separate test set.

## WSI and patch contract

| Item | Specification |
| --- | --- |
| Original WSI | KF-PRO-005-EX scans acquired with a 40x objective; private KFB files are not redistributed |
| WSI conversion | Convert KFB to TIFF with [KFbioConverter](https://github.com/WilmerWang/SLFCD/releases) before OpenSlide reading |
| WSI formats | `.svs`, `.tif`, `.tiff`, `.ndpi`, `.mrxs` are supported by the feature extractors |
| WSI resizing | The complete WSI is not resized. Its native pyramid and dimensions are preserved |
| Patch coordinates | Read from the `coords` dataset in an H5 file |
| Patch size | Non-overlapping `512 x 512` pixels at the maximum-resolution level (`patch_level=0`) |
| Pyramid level | Read from the H5 `patch_level` attribute; the reported workflow uses the maximum-resolution level |
| Downsampling | `custom_downsample=1` unless recorded otherwise in the H5 attributes |
| Color space | RGB after OpenSlide/PIL conversion |
| Stain normalization | Macenko normalization is applied inside every feature extractor before the encoder transform |
| Tissue segmentation | Follow the public CLAM segmentation and patching workflow; the restricted local preprocessing copy is not included |

The physical magnification or effective microns-per-pixel is determined by the
WSI pyramid and `patch_level`. It is not safe to claim a single 10x/20x value
without the original H5 attributes and WSI metadata. The UNI2-h extractor
records `base_mpp`, `effective_mpp`, and `level_dimensions` when OpenSlide
provides them.

## Patch-to-model preprocessing

| Extractor | Patch read from WSI | Model input | Normalization / stain processing | Output |
| --- | --- | --- | --- | --- |
| MUSK | `512 x 512` | Resize/center-crop to `384 x 384` | Macenko, then RGB tensor mean/std `(0.5, 0.5, 0.5)`; global image embedding, no projection or output normalization; `ms_aug=False` | `1024-D` features used by the main MIL configs |
| UNI2-h | `512 x 512` | Official `224 x 224` transform | Macenko, then timm-resolved official transform; fallback uses ImageNet mean/std `(0.485, 0.456, 0.406)` and `(0.229, 0.224, 0.225)` | `1536-D` features |
| CONCH | `512 x 512` | Official transform returned by the CONCH loader | Macenko, then official CONCH transform; `proj_contrast=False`, `normalize=False` | Detected from the encoder and written to metadata |
| ImageNet ResNet50 | `512 x 512` | Official torchvision `ResNet50_Weights.IMAGENET1K_V2` transform | Macenko, then ImageNet normalization from the torchvision weight object; classifier head removed | `2048-D` |

The MUSK extractor uses `musk_large_patch16_384` with `ms_aug=False`. The UNI2-h architecture is
`vit_giant_patch14_224` with depth `24`, `24` heads, embedding dimension
`1536`, `8` register tokens, and `dynamic_img_size=True`. The CONCH extractor
uses the official `conch_vitb` loader and its official preprocessing transform.

## Downstream training parameters

The reported main configuration uses five internal folds, batch size `1`,
`1024-D` input features, three slide classes (`CD=1`, `ITB=0`, `NO=2`), two
case classes (`CD=1`, `ITB=0`), `100` epochs, validation every epoch, and
four data-loader workers. The training entry points use seed `421` for the
reported runs. The main slide model uses `SRMamba`, `2` layers, dropout `0.3`,
and rate `10`; the case head uses a two-layer Transformer with `8` heads,
MLP ratio `4.0`, and dropout `0.3`.

TransMIL receives `1024-D` features and projects them to a `512-D` slide
embedding before the case-level head. For UNI2-h or CONCH features, update
`DATA.hidden_dim` to the actual feature dimension before training.

## Upstream preprocessing references

- [CLAM official repository](https://github.com/mahmoodlab/CLAM)
- [CLAM processing documentation](https://github.com/mahmoodlab/CLAM/blob/master/docs/README.md)
- [OpenSlide](https://openslide.org/)
- [torchstain Macenko implementation](https://github.com/EIDOSLAB/torchstain)
- [torchvision ResNet50](https://pytorch.org/vision/stable/models/resnet.html)

The shared implementation is `feature_extraction/common/stain.py`. It fits
Macenko target statistics from the first valid tissue patch in the current
extraction run and reuses that normalizer. A laboratory with a fixed reference
patch can pass one to `MacenkoStainNormalizer` without changing the downstream
format. The project follows CLAM for tissue segmentation and patch-coordinate
generation, but the private local WSI preprocessing files are intentionally not
redistributed.
