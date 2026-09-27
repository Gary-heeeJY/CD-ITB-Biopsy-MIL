"""Extract frozen ImageNet ResNet50 features from CLAM-style WSI patches.

The extractor consumes a directory of WSI files and a directory of H5 files
containing ``coords`` plus ``patch_level`` and ``patch_size`` attributes. Each
slide is written as ``<output>/<slide_id>/image_feature.pt`` together with
patch metadata, matching the downstream MIL loader in this repository.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import h5py
import openslide
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from config import Config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.stain import MacenkoStainNormalizer


LOGGER = logging.getLogger("resnet50_features")


def _device(value: str) -> torch.device:
    if value == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if value.startswith("cuda:") and torch.cuda.is_available():
        return torch.device(value)
    return torch.device("cpu")


def _load_model(weights_path: str | None, device: torch.device):
    if weights_path:
        model = resnet50(weights=None)
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        weights_name = str(weights_path)
        transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    else:
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        weights_name = weights.name
        transform = weights.transforms()

    model.fc = torch.nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, transform, weights_name


def _read_h5(path: Path):
    with h5py.File(path, "r") as handle:
        if "coords" not in handle:
            raise KeyError(f"Missing coords dataset: {path}")
        coords = handle["coords"][:]
        attrs = handle["coords"].attrs
        patch_level = int(attrs.get("patch_level", 0))
        patch_size = int(attrs.get("patch_size", 512))
        custom_downsample = float(attrs.get("custom_downsample", 1.0))
    return coords, patch_level, patch_size, custom_downsample


def _slide_ids(wsi_dir: Path, patch_dir: Path, extensions: Iterable[str]):
    for wsi_path in sorted(wsi_dir.iterdir()):
        if not wsi_path.is_file() or wsi_path.suffix.lower() not in extensions:
            continue
        h5_path = patch_dir / f"{wsi_path.stem}.h5"
        if h5_path.is_file():
            yield wsi_path, h5_path
        else:
            LOGGER.warning("Skipping %s because %s is missing", wsi_path.name, h5_path.name)


def extract_slide(model, transform, stain_normalizer, device, wsi_path: Path, h5_path: Path, output_dir: Path, batch_size: int, weights_name: str):
    coords, patch_level, patch_size, custom_downsample = _read_h5(h5_path)
    features = []
    patch_metadata = []
    slide = openslide.open_slide(str(wsi_path))
    try:
        for start in range(0, len(coords), batch_size):
            batch_images = []
            batch_metadata = []
            for coord in coords[start : start + batch_size]:
                try:
                    image = slide.read_region(
                        (int(coord[0]), int(coord[1])),
                        patch_level,
                        (patch_size, patch_size),
                    ).convert("RGB")
                    image = stain_normalizer(image)
                    batch_images.append(transform(image))
                    batch_metadata.append({
                        "coord": [int(coord[0]), int(coord[1])],
                        "patch_level": patch_level,
                        "patch_size": patch_size,
                        "custom_downsample": custom_downsample,
                    })
                except Exception:
                    LOGGER.exception("Failed to read patch %s from %s", coord, wsi_path.name)

            if not batch_images:
                continue

            batch = torch.stack(batch_images).to(device, non_blocking=True)
            with torch.inference_mode():
                batch_features = model(batch).float().cpu()
            features.append(batch_features)
            patch_metadata.extend(batch_metadata)
    finally:
        slide.close()

    if features:
        feature_tensor = torch.cat(features, dim=0)
    else:
        feature_tensor = torch.empty((0, 2048), dtype=torch.float32)

    if feature_tensor.shape[0] != len(patch_metadata):
        raise RuntimeError(f"Feature/metadata mismatch for {wsi_path.name}")

    slide_output = output_dir / wsi_path.stem
    slide_output.mkdir(parents=True, exist_ok=True)
    torch.save(feature_tensor, slide_output / "image_feature.pt")
    (slide_output / "patch_metadata.json").write_text(
        json.dumps(patch_metadata, indent=2), encoding="utf-8"
    )
    metadata = {
        "slide_id": wsi_path.stem,
        "feature_extractor": "torchvision_resnet50",
        "weights": weights_name,
        "feature_dim": 2048,
        "n_patches_total": int(len(coords)),
        "n_patches_used": int(feature_tensor.shape[0]),
        "patch_level": patch_level,
        "patch_size": patch_size,
        "custom_downsample": custom_downsample,
        "stain_normalization": "Macenko",
    }
    (slide_output / "feature_extractor_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    LOGGER.info("%s: saved %s", wsi_path.name, tuple(feature_tensor.shape))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi-dir", type=Path, default=Config.WSI_DIR)
    parser.add_argument("--patch-dir", type=Path, default=Config.PATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=Config.FEATURE_DIR)
    parser.add_argument("--weights", type=str, default=Config.WEIGHTS_PATH,
                        help="Optional local ResNet50 checkpoint; otherwise torchvision downloads ImageNet weights.")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--device", default=Config.DEVICE, help="cpu, cuda, or cuda:N")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    torch.manual_seed(Config.SEED)
    device = _device(args.device)
    model, transform, weights_name = _load_model(args.weights, device)
    stain_normalizer = MacenkoStainNormalizer()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(_slide_ids(args.wsi_dir, args.patch_dir, Config.SLIDE_EXTENSIONS))
    if not pairs:
        raise SystemExit("No matching WSI/H5 pairs found. Check --wsi-dir and --patch-dir.")

    for wsi_path, h5_path in pairs:
        output_file = args.output_dir / wsi_path.stem / "image_feature.pt"
        if output_file.exists() and not args.overwrite:
            LOGGER.info("Skipping existing output: %s", output_file)
            continue
        extract_slide(model, transform, stain_normalizer, device, wsi_path, h5_path, args.output_dir, args.batch_size, weights_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
