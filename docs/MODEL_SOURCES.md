# Model Code and Weight Sources

The models below are used as frozen feature extractors in this project. This
repository does not redistribute their checkpoints. Users must follow each
upstream project's access, license, and citation requirements.

| Component | Role in this project | Official code | Official weights / access |
| --- | --- | --- | --- |
| CLAM | WSI tissue segmentation and patch-coordinate workflow reference | [MahmoodLab/CLAM](https://github.com/mahmoodlab/CLAM) | CLAM preprocessing does not require a project-specific foundation-model checkpoint |
| MUSK | Patch-level frozen vision-language feature extractor | [lilab-stanford/MUSK](https://github.com/lilab-stanford/MUSK) | [xiangjx/musk on Hugging Face](https://huggingface.co/xiangjx/musk) |
| UNI2-h | Patch-level frozen pathology feature extractor | [MahmoodLab/UNI](https://github.com/mahmoodlab/UNI) | [MahmoodLab/UNI2-h on Hugging Face](https://huggingface.co/MahmoodLab/UNI2-h) |
| CONCH | Patch-level frozen pathology vision-language feature extractor | [MahmoodLab/CONCH](https://github.com/mahmoodlab/CONCH) | [MahmoodLab/CONCH on Hugging Face](https://huggingface.co/MahmoodLab/CONCH) |
| ResNet50 | ImageNet feature-extraction comparison | [torchvision models](https://pytorch.org/vision/stable/models/resnet.html) | [ResNet50 weights](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) |
| Macenko | Required stain normalization before every frozen encoder | [EIDOSLAB/torchstain](https://github.com/EIDOSLAB/torchstain) | No learned weights; the shared implementation is `feature_extraction/common/stain.py` |
| TransMIL | Reported direct MIL comparison | [szc19990412/TransMIL](https://github.com/szc19990412/TransMIL) | No pretrained weight is used by this project for downstream training |
| MambaMIL | Main slide-level MIL backbone and direct baseline reference | [isyangshu/MambaMIL](https://github.com/isyangshu/MambaMIL) | No pretrained weight is used by this project for downstream training |

## Access notes

MUSK, UNI/UNI2-h, and CONCH weights may require accepting the authors'
academic-use terms and authenticating with Hugging Face. The links above are
the authoritative starting points for access and usage instructions. Do not
commit downloaded checkpoints to this repository.

## What is actually used here

- No foundation model is fine-tuned in the reported experiments.
- MUSK, UNI2-h, CONCH, and ImageNet ResNet50 are used as frozen encoders.
- The repository includes local reference/adaptation scripts under
  `feature_extraction/`, but it does not claim to replace the upstream
  implementations.
- The public repository now includes
  `feature_extraction/resnet50/extract_features_resnet50.py`, which creates
  ImageNet ResNet50 features from the same CLAM-compatible WSI/H5 contract.
  The original private feature tensors are still not distributed.
