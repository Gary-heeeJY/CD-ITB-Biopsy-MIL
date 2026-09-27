# ResNet50 Feature Extraction

This entry point creates the ImageNet-pretrained ResNet50 comparison features
used by the downstream MIL code. It follows the CLAM-compatible H5 contract:
`coords`, `patch_level`, `patch_size`, and optional `custom_downsample`.
The reported workflow uses non-overlapping 512 x 512 patches at
`patch_level=0`; Macenko stain normalization is applied inside this script
before the official ImageNet transform.

```bash
python feature_extraction/resnet50/extract_features_resnet50.py \
  --wsi-dir data/wsi \
  --patch-dir data/patches \
  --output-dir data/features/ResNet50
```

By default, torchvision downloads `ResNet50_Weights.IMAGENET1K_V2`. To use a
local checkpoint instead:

```bash
python feature_extraction/resnet50/extract_features_resnet50.py \
  --weights weights/ResNet50/resnet50-11ad3fa6.pth
```

Each slide is saved as:

```text
data/features/ResNet50/<slide_id>/
  image_feature.pt
  patch_metadata.json
  feature_extractor_metadata.json
```

The output feature dimension is 2048. Raw WSIs, H5 files, and weights are
local inputs and must not be committed.
