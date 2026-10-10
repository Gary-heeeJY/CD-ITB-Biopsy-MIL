# CD-ITB Hierarchical MambaMIL

This repository contains the code for the paper on differentiating Crohn's
disease (CD) from intestinal tuberculosis (ITB) using biopsy whole-slide
images. The public release is organized around the experiments reported in the
paper and starts from either CLAM-compatible patches or pre-extracted
patch-level features.

The release layout follows the practical structure used by public WSI
reproduction repositories such as [XpressO](https://github.com/skr1/XpressO):
data preparation, feature extraction, classification, evaluation, and
heatmaps are separate documented stages.

Raw WSIs, patient identifiers, private labels and splits, extracted feature
tensors, checkpoints, and foundation-model weights are intentionally excluded.
The repository provides the data contract, relative demo paths, model
configuration, feature-extraction references, and reproducible downstream
training/testing entry points.

## Repository layout

```text
configs/             Reported experiment configurations
data/                Public data/feature contract; no private data
docs/                Reproduction, preprocessing, and model-source notes
feature_extraction/  MUSK, UNI2-h, CONCH, and ResNet50 extractors
framework/           MIL models, datasets, losses, engines, and heatmaps
scripts/             Stable public training/testing entry points
tests/               Dependency-light release checks
```

The repository includes a [CITATION.cff](CITATION.cff) file for software
citation.

The label convention is fixed throughout the project: CD is the positive
class (`1`), ITB/TB is the negative class (`0`), and NO is the slide-level
class (`2`). All reported runs use random seed `421`.

## Environment

The code was organized for Python 3.10. Install a PyTorch build compatible
with the target CUDA runtime first, then install the remaining dependencies:

```bash
conda create -n cd_itb_mil python=3.10 -y
conda activate cd_itb_mil
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

The Mamba dependencies must match the selected PyTorch/CUDA build. OpenSlide
also requires its system library; follow the installation instructions for
your operating system before running WSI feature extraction.

## Data contract

Create local data using the following relative layout. These directories are
examples only and are ignored by Git:

```text
data/
  labels/
    internal_train.csv
    internal_test.csv
    external_test.csv
  splits/
    internal_train/
      splits_0.csv ... splits_4.csv
  wsi/                         # optional, needed for extraction/heatmaps
  patches/                     # optional, CLAM-generated H5 files
  features/
    MUSK/
    UNI2-h/
    CONCH/
    ResNet50/
    MUSK_CORAL/                 # optional external-only aligned features
    UNI2-h_CORAL/
    CONCH_CORAL/
    ResNet50_CORAL/
```

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for CSV columns, H5
attributes, feature-file formats, and the expected slide directory structure.
The final pre-push checks are listed in
[docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md).

## Preprocessing and feature extraction

The private WSI segmentation copy is not distributed. Use the public
[CLAM workflow](https://github.com/mahmoodlab/CLAM) to create tissue masks,
patch coordinates, and H5 files. The exact public contract used by this
project is documented in [docs/PREPROCESSING.md](docs/PREPROCESSING.md).

The foundation models are frozen. Use the authors' public code and weights;
the access links are collected in [docs/MODEL_SOURCES.md](docs/MODEL_SOURCES.md).

For example, ResNet50 features can be generated end to end with:

```bash
python feature_extraction/resnet50/extract_features_resnet50.py \
  --wsi-dir data/wsi \
  --patch-dir data/patches \
  --output-dir data/features/ResNet50
```

The other extractors are documented in their local READMEs:

```bash
python feature_extraction/musk/extract_features_musk.py
python feature_extraction/uni/extract_features_uni2-h.py
python feature_extraction/conch/extract_features_conch.py
```

All four commands include Macenko stain normalization internally. There is no
separate Macenko variant to run. The common implementation is
`feature_extraction/common/stain.py` and records `stain_normalization` in the
feature metadata.

## CORAL external alignment

The paper's CORAL step is documented in [docs/CORAL.md](docs/CORAL.md). It is
an external-inference preprocessing step: estimate target statistics from the
internal training cohort, estimate source statistics from each external slide
alone, and write aligned external features to a new directory. Never fit CORAL
using external labels or pooled external test statistics.

Edit only the relative paths and local weight paths in each extractor config.
Do not commit downloaded weights or generated feature tensors.

## Reported experiments

Train the proposed hierarchical model:

```bash
python scripts/train_main.py --cfg configs/MambaMIL_DynamicWeight.yaml
```

The reported frozen-encoder comparison uses the same proposed model with
encoder-specific feature dimensions:

```bash
python scripts/train_main.py --cfg configs/MambaMIL_Encoder_ResNet50.yaml
python scripts/train_main.py --cfg configs/MambaMIL_Encoder_CONCH.yaml
python scripts/train_main.py --cfg configs/MambaMIL_Encoder_UNI2-h.yaml
```

Evaluate it on the external independent test set after creating
`data/features/MUSK_CORAL` with the CORAL command below. Supply the local
labels, aligned features, and trained output directory explicitly:

```bash
python scripts/test_main.py --cfg configs/MambaMIL_DynamicWeight.yaml \
  --expe_name outputs/AJP_Comparison_experiments/MUSK_MambaMIL_DynamicWeight/seed-421 \
  --ex_result outputs/main_external_test \
  --label_csv_file data/labels/external_test.csv \
  --feat_file data/features/MUSK_CORAL
```

Run the reported direct patch-to-patient baselines:

```bash
python scripts/train_baselines.py --models ABMIL,TransMIL,MambaMIL
python scripts/test_baselines.py --models ABMIL,TransMIL,MambaMIL
```

Run the reported AGTA ablation:

```bash
python scripts/train_ablation.py --variant meanpool
```

To separate the two internal components of the case-level aggregation,
train the two additional controls with the same slide-level MambaMIL,
features, splits, optimizer, and seed:

```bash
python scripts/train_ablation.py --variant attention_only
python scripts/train_ablation.py --variant transformer_only
```

The four settings are therefore: `meanpool` (A), `attention_only` (B),
`transformer_only` (C), and the proposed full model (D). B keeps the gated
attention-based feature selection and removes the Transformer. C keeps the
Transformer and uses a learned case token instead of the attention-derived
token. The full model is trained with:

```bash
python scripts/train_main.py --cfg configs/MambaMIL_DynamicWeight.yaml
```

After each training run, evaluate the corresponding checkpoint on the
internal or external labels and features. The wrapper supplies the correct
case-level model automatically; only paths that point to your local data and
checkpoint directory need to be changed:

```bash
python scripts/test_ablation.py --variant meanpool \
  --expe_name outputs/AJP_Comparison_experiments/MUSK_MambaMIL_DynamicWeight_A_meanpool/seed-421 \
  --ex_result outputs/ablation_meanpool_internal \
  --label_csv_file data/labels/internal_test.csv \
  --feat_file data/features/MUSK

python scripts/test_ablation.py --variant attention_only \
  --expe_name outputs/AJP_Comparison_experiments/MUSK_MambaMIL_DynamicWeight_B_attention_only/seed-421 \
  --ex_result outputs/ablation_attention_only_internal \
  --label_csv_file data/labels/internal_test.csv \
  --feat_file data/features/MUSK

python scripts/test_ablation.py --variant transformer_only \
  --expe_name outputs/AJP_Comparison_experiments/MUSK_MambaMIL_DynamicWeight_C_transformer_only/seed-421 \
  --ex_result outputs/ablation_transformer_only_internal \
  --label_csv_file data/labels/internal_test.csv \
  --feat_file data/features/MUSK
```

For external evaluation, replace the label file and feature directory with
the external set and its CORAL-aligned features, for example
`data/features/MUSK_CORAL`. Each training run writes a pooled out-of-fold
validation threshold. The test entry points load that file and never fit a
threshold from internal or external test labels.

The binary CD-versus-ITB threshold is fitted once from pooled out-of-fold
validation predictions and saved as `validation_threshold.json`. Test scripts
load that threshold and never fit it using test labels.

## Heatmaps

Heatmaps require local WSI files, matching H5 coordinates, matching feature
files, and a trained checkpoint:

```bash
python scripts/make_heatmaps.py --help
```

The full command map is in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Upstream projects and citations

This repository adapts or calls the following public projects: CLAM, MUSK,
UNI/UNI2-h, CONCH, torchvision ResNet50, torchstain, TransMIL, and MambaMIL.
Their official code and weight links, license notes, and citation reminders
are listed in [docs/MODEL_SOURCES.md](docs/MODEL_SOURCES.md). Please retain
their licenses and cite the corresponding papers when publishing results.

## Release policy

Before pushing to GitHub, confirm that `data/`, `weights/`, `outputs/`, and
generated result files are empty or ignored, then run:

```bash
python scripts/validate_release.py
python -m compileall -q framework scripts feature_extraction
```

The repository does not grant access to private patient data. Reproduction of
the reported numerical results requires the authors' matching feature files
and split files or an approved replacement dataset.
