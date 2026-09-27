"""Validate the public source tree without importing ML dependencies."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "requirements.txt",
    "configs/MambaMIL_DynamicWeight.yaml",
    "configs/MambaMIL.yaml",
    "configs/TransMIL.yaml",
    "configs/biospy_case_abmil.yaml",
    "configs/MambaMIL_Encoder_ResNet50.yaml",
    "configs/MambaMIL_Encoder_CONCH.yaml",
    "configs/MambaMIL_Encoder_UNI2-h.yaml",
    "scripts/train_main.py",
    "scripts/test_main.py",
    "scripts/train_baselines.py",
    "scripts/test_baselines.py",
    "scripts/train_ablation.py",
    "scripts/test_ablation.py",
    "scripts/make_heatmaps.py",
    "docs/DATA_FORMAT.md",
    "docs/REPRODUCE.md",
    "docs/PREPROCESSING.md",
    "docs/MODEL_SOURCES.md",
    "docs/CORAL.md",
    "feature_extraction/common/stain.py",
    "feature_extraction/resnet50/extract_features_resnet50.py",
)
MARKERS = (
    "/nas_3",
    "/server03_data",
    "newnas_1",
    "api_key=\"",
    "65ej",
    "<LOCAL_DATA_PATH>",
    "<PATCH_ROOT>",
    "<WSI_ROOT>",
    "ms_aug=True",
)


def main() -> int:
    missing = [item for item in REQUIRED if not (ROOT / item).is_file()]
    if missing:
        print("Missing required files:")
        print("\n".join(missing))
        return 1

    candidates = []
    for pattern in ("*.py", "*.yaml", "*.yml", "*.md"):
        candidates.extend(ROOT.rglob(pattern))
    hits = []
    for path in candidates:
        if (
            path.name == "validate_release.py"
            or "docs/superpowers" in path.as_posix()
            or "tests" in path.parts
        ):
            continue
        content = path.read_text(encoding="utf-8")
        for marker in MARKERS:
            if marker in content:
                hits.append(f"{path.relative_to(ROOT)}: {marker}")
    if hits:
        print("Private markers found:")
        print("\n".join(hits))
        return 1

    print(f"Public release validation passed: {len(REQUIRED)} required files checked")
    print(f"Scanned {len(candidates)} public scripts/configs for private markers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
