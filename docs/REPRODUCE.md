# Reproduce the reported experiments

This project reproduces downstream MIL experiments from pre-extracted
patch-level features. Raw WSIs and private labels/splits are not included.
The complete WSI-to-feature contract is described in
[PREPROCESSING.md](PREPROCESSING.md).

## 1. Prepare local data

Follow [DATA_FORMAT.md](DATA_FORMAT.md). Put the matching feature directory in
`data/features/` and update the config only when using a different extractor.

If reproducing feature extraction, first follow the public CLAM workflow for
tissue segmentation and non-overlapping 512 x 512 patches at maximum
resolution. Then run exactly one extractor per encoder:

```bash
python feature_extraction/musk/extract_features_musk.py
python feature_extraction/uni/extract_features_uni2-h.py
python feature_extraction/conch/extract_features_conch.py
python feature_extraction/resnet50/extract_features_resnet50.py
```

Each command applies Macenko internally. No separate Macenko script is used.

## 2. Main hierarchical model

The main config uses frozen 1024-dimensional MUSK features, five internal
folds, seed 421, and the reported hierarchical MambaMIL model:

```bash
python scripts/train_main.py --cfg configs/MambaMIL_DynamicWeight.yaml
```

Feature-encoder comparison (same hierarchical model, frozen encoder features):

```bash
python scripts/train_main.py --cfg configs/MambaMIL_Encoder_ResNet50.yaml
python scripts/train_main.py --cfg configs/MambaMIL_Encoder_CONCH.yaml
python scripts/train_main.py --cfg configs/MambaMIL_Encoder_UNI2-h.yaml
```

The manuscript labels the last comparison as UNI; this repository uses the
UNI2-h implementation and names it explicitly to avoid ambiguity.

For an external evaluation that follows the paper's CORAL protocol, create an
external-only aligned feature directory first:

```bash
python scripts/apply_coral.py \
  --feature-root data/features/MUSK \
  --output-dir data/features/MUSK_CORAL
```

Then evaluate on the external independent test set:

```bash
python scripts/test_main.py --cfg configs/MambaMIL_DynamicWeight.yaml \
  --expe_name outputs/AJP_Comparison_experiments/<task>/seed-421 \
  --ex_result outputs/main_test \
  --label_csv_file data/labels/external_test.csv \
  --feat_file data/features/MUSK_CORAL
```

Repeat with `UNI2-h`, `CONCH`, or `ResNet50` for the encoder comparison. See
[CORAL.md](CORAL.md) for the full protocol and commands.

## 3. Direct baselines

These models map all patch features directly to the case/patient label:

```bash
python scripts/train_baselines.py --models ABMIL,TransMIL,MambaMIL
python scripts/test_baselines.py --models ABMIL,TransMIL,MambaMIL
```

## 4. Reported ablations

```bash
python scripts/train_ablation.py --variant meanpool
python scripts/train_ablation.py --variant attention_only
python scripts/train_ablation.py --variant transformer_only
```

The four settings use the same slide-level MambaMIL training pipeline:

* A: mean pooling, without gated attention or a Transformer.
* B: gated attention only, without a Transformer.
* C: Transformer only, with a learned case token instead of the attention
  token.
* D: the proposed gated-attention plus Transformer aggregation.

Evaluate B and C with the matching public test entry points:

```bash
python scripts/test_ablation.py --variant attention_only \
  --expe_name outputs/AJP_Comparison_experiments/MUSK_MambaMIL_DynamicWeight_B_attention_only/seed-421 \
  --ex_result outputs/ablation_attention_only_external \
  --label_csv_file data/labels/external_test.csv \
  --feat_file data/features/MUSK_CORAL

python scripts/test_ablation.py --variant transformer_only \
  --expe_name outputs/AJP_Comparison_experiments/MUSK_MambaMIL_DynamicWeight_C_transformer_only/seed-421 \
  --ex_result outputs/ablation_transformer_only_external \
  --label_csv_file data/labels/external_test.csv \
  --feat_file data/features/MUSK_CORAL
```

Max-pooling, CLAM comparison, and PEGTF-MIL comparison are not part of the
reported experiment set. TransMIL remains available as a direct
patch-to-patient baseline through `scripts/train_baselines.py`.

## 5. Threshold protocol

CD is the positive class. AUC is calculated from CD probability and is
threshold-independent. Binary accuracy, sensitivity, and specificity use one
threshold fitted by maximizing the Youden index on pooled out-of-fold
validation predictions. Training writes this value to
`validation_threshold.json`; external-test code loads it and never uses
external-test labels to select a threshold.

## 6. Heatmaps

Provide local WSI files, H5 patch coordinates, matching feature files, and a
trained checkpoint:

```bash
python scripts/make_heatmaps.py --help
```

The heatmap code is an interpretation utility and does not alter training or
test metrics.
