# Local data layout

The public repository does not contain patient data, WSIs, labels, split
files, feature tensors, or checkpoints. Create the following local layout:

```text
data/
  labels/
    internal_train.csv
    internal_test.csv
    external_test.csv
  splits/
    internal_train/
      splits_0.csv
      splits_1.csv
      splits_2.csv
      splits_3.csv
      splits_4.csv
  wsi/
  patches/
  features/
    MUSK/<slide_id>/image_feature.pt
    UNI2-h/<slide_id>/image_feature.pt
    CONCH/<slide_id>/image_feature.pt
    ResNet50/<slide_id>/image_feature.pt
    MUSK_CORAL/<slide_id>/image_feature.pt
    UNI2-h_CORAL/<slide_id>/image_feature.pt
    CONCH_CORAL/<slide_id>/image_feature.pt
    ResNet50_CORAL/<slide_id>/image_feature.pt
```

The `wsi/` and `patches/` directories are needed only for feature extraction
or heatmap generation. Downstream MIL training starts from `features/`.

Each label CSV uses:

```text
slide_id,case_id,slide_label,case_label
CD_Case001_01,Case001,CD,CD
ITB_Case002_01,Case002,ITB,ITB
NO_Case003_01,Case003,NO,
```

The five split files under `splits/internal_train/` contain `train`, `val`,
and `test` slide-ID columns. All slides from a patient must remain in one
fold. H5 files are CLAM-compatible and contain `coords` with `patch_level=0`,
`patch_size=512`, and usually `custom_downsample=1`. The WSI basename, H5
basename, label `slide_id`, and feature directory name must be identical.

`*_CORAL` directories are generated only for external inference. They use
internal-training statistics as the target and statistics from each external
slide as its source; they must not replace ordinary training features.

See [docs/DATA_FORMAT.md](../docs/DATA_FORMAT.md) and
[docs/PREPROCESSING.md](../docs/PREPROCESSING.md) for the required file
formats and preprocessing parameters.
