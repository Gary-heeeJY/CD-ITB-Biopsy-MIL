# CORAL External Alignment

CORAL (Correlation Alignment) is used only when evaluating on the external
cohort. It is applied to patch-level frozen-encoder features before the normal
MIL test code.

## Protocol used by the paper

1. Read patch features from the internal training cohort and estimate one
   target mean and covariance.
2. For each external slide independently, estimate source statistics from that
   slide's own patches.
3. Whiten the external slide with its own statistics and recolor it with the
   internal-training statistics.
4. Save the aligned features in a new directory. Internal training,
   validation, and internal independent-test features are not overwritten.

The implementation uses seed `421`, covariance shrinkage `0.10`, and diagonal
regularization `1e-5`. It does not use external labels and never pools patches
from different external slides to estimate the source covariance.

## Commands

Run from the repository root after generating the ordinary features:

```bash
python scripts/apply_coral.py \
  --feature-root data/features/MUSK \
  --output-dir data/features/MUSK_CORAL

python scripts/apply_coral.py \
  --feature-root data/features/UNI2-h \
  --output-dir data/features/UNI2-h_CORAL

python scripts/apply_coral.py \
  --feature-root data/features/CONCH \
  --output-dir data/features/CONCH_CORAL

python scripts/apply_coral.py \
  --feature-root data/features/ResNet50 \
  --output-dir data/features/ResNet50_CORAL
```

Then point the external test command at the corresponding `_CORAL` directory.
The output contains the original metadata plus `coral_metadata.json`, which
records that the source statistics were estimated per external slide.

`framework/utils_/Coral.py` is retained only as a MUSK compatibility wrapper.
New reproductions should use `scripts/apply_coral.py` so MUSK, UNI2-h, CONCH,
and ResNet50 follow the same paper-faithful protocol.
 the same
paper-faithful protocol.

