# Model weights

Weights are not committed to this repository. Download them from the
official sources listed in [MODEL_SOURCES.md](MODEL_SOURCES.md), observe each
project's license and access requirements, and place local copies under
`weights/`.

Expected demo locations are:

```text
weights/
  MUSK/musk.pth
  UNI2-h/pytorch_model.bin
  CONCH/conch_checkpoint.pt
  ResNet50/resnet50-11ad3fa6.pth
```

All foundation encoders are frozen during feature extraction. The downstream
MIL models are trained from scratch for the reported experiments.
