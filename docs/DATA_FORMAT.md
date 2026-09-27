# Data and feature format

## Dataset layout

The layout below mirrors the experiment stages. Labels, patient metadata,
features, and WSIs are intentionally absent from Git because the source data
are restricted.

```text
data/
  labels/
    internal_train.csv       # Center 1 training/validation cohort
    internal_test.csv        # Center 1 independent test cohort
    external_test.csv        # Center 2 independent test cohort
  splits/
    internal_train/
      splits_0.csv ... splits_4.csv
  wsi/                        # optional; needed for feature extraction/heatmaps
    <slide_id>.tif
  patches/                    # optional; CLAM-generated H5 files
    <slide_id>.h5
  features/
    MUSK/<slide_id>/image_feature.pt
    UNI2-h/<slide_id>/image_feature.pt
    CONCH/<slide_id>/image_feature.pt
    ResNet50/<slide_id>/image_feature.pt
    MUSK_CORAL/               # optional external-only aligned features
    UNI2-h_CORAL/
    CONCH_CORAL/
    ResNet50_CORAL/
```

The same `<slide_id>` must be used in the label CSV, WSI filename, H5 filename,
feature directory, and `patch_metadata.json`. Do not use patient identifiers as
slide identifiers when one patient has multiple WSIs.

## Labels and splits

The label CSV must contain the identifiers needed by the selected loader:

```text
slide_id,case_id,slide_label,case_label
CD_Case001_01,Case001,CD,CD
ITB_Case002_01,Case002,ITB,ITB
NO_Case003_01,Case003,NO,
```

`slide_label` is the three-class auxiliary label (`CD`, `ITB`, or `NO`).
`case_label` is the binary patient label (`CD` or `ITB`) and is blank for the
normal control rows. The loaders convert these strings to model labels:
CD=`1`, ITB=`0`, and NO=`2` at slide level; CD=`1` and ITB=`0` at case level.

Each `splits_<k>.csv` contains the slide IDs for one patient-level fold. The
split loader expects the columns `train`, `val`, and `test`; every value must be
a slide ID from the corresponding labels file. All slides from one patient
must stay in one fold.

The public label convention is CD=`1`, ITB/TB=`0`, and NO=`2` for slide-level
three-class data. The case-level task uses CD=`1` and ITB/TB=`0`.

## H5 patch coordinates

Each CLAM-compatible H5 file contains a `coords` dataset. Its attributes
should include:

```text
patch_level: integer WSI pyramid level, 0 for the reported maximum-resolution workflow
patch_size: integer patch edge length, 512 for the reported workflow
custom_downsample: optional numeric downsampling factor, normally 1
```

The feature extractors read these values instead of assuming a fixed WSI
magnification. The public WSI workflow reference is
[CLAM](https://github.com/mahmoodlab/CLAM).

## Feature files

Each slide should be stored as a directory:

```text
data/features/MUSK/<slide_id>/
  image_feature.pt
  patch_metadata.json              # coordinates aligned row-for-row with features
  feature_extractor_metadata.json  # encoder, dimensions, patch and stain settings
```

`image_feature.pt` may be a tensor shaped `[n_patches, feature_dim]` or a
dictionary containing `features` and optional `coords`. The loader also
accepts a flat `<slide_id>.pt` file.

Expected feature dimensions are:

| Extractor | Dimension | Main use |
| --- | ---: | --- |
| MUSK | 1024 | Main reported hierarchical MIL run |
| UNI2-h | 1536 | Foundation-model comparison/reference |
| CONCH | recorded in metadata | Feature extraction reference; set `DATA.hidden_dim` accordingly |
| ImageNet ResNet50 | 2048 | ResNet50 comparison |

The selected config's `DATA.hidden_dim` must match the feature tensors.
