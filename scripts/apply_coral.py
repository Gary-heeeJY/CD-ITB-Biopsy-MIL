"""Apply leakage-safe CORAL alignment to an external feature cohort.

The internal training cohort supplies the target mean/covariance. Each
external slide supplies its own source mean/covariance, so no pooled external
test statistic or external label is used. The output keeps the same slide
directory layout and can be passed to the normal test entry point.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_features(path: Path) -> tuple[torch.Tensor, dict | None]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older PyTorch versions
        data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        features = data["features"]
        payload = data.copy()
    else:
        features = data
        payload = None
    if features.ndim == 1:
        features = features.unsqueeze(0)
    return features.float(), payload


def find_feature(root: Path, slide_id: str, case_id: str | None = None) -> tuple[Path | None, Path | None]:
    candidates = [slide_id]
    if case_id and case_id not in candidates:
        candidates.append(case_id)
    for candidate in candidates:
        folder = root / candidate
        if folder.is_dir() and (folder / "image_feature.pt").is_file():
            return folder / "image_feature.pt", folder
        flat = root / f"{candidate}.pt"
        if flat.is_file():
            return flat, root
    matches = sorted(glob.glob(str(root / f"{slide_id}*")))
    for match in matches:
        folder = Path(match)
        if folder.is_dir() and (folder / "image_feature.pt").is_file():
            return folder / "image_feature.pt", folder
    return None, None


def regularized_covariance(features: np.ndarray, shrinkage: float, eps: float) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    if features.shape[0] <= 2:
        variance = features.var(axis=0) + eps
        covariance = np.diag(variance)
    else:
        covariance = np.cov(features, rowvar=False)
        diagonal = np.diag(np.diag(covariance))
        covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
        covariance = covariance + np.eye(covariance.shape[0]) * eps
    return mean.astype(np.float64), covariance.astype(np.float64)


def collect_internal_stats(csv_path: Path, root: Path, max_patches: int, sample_rate: float, seed: int, shrinkage: float, eps: float):
    generator = torch.Generator().manual_seed(seed)
    chunks = []
    collected = 0
    table = pd.read_csv(csv_path)
    for _, row in table.iterrows():
        if collected >= max_patches:
            break
        feature_path, _ = find_feature(root, str(row["slide_id"]), str(row["case_id"]) if "case_id" in row else None)
        if feature_path is None:
            continue
        features, _ = load_features(feature_path)
        n = max(1, int(features.shape[0] * sample_rate))
        if n < features.shape[0]:
            indices = torch.randperm(features.shape[0], generator=generator)[:n]
            features = features[indices]
        remaining = max_patches - collected
        features = features[:remaining]
        chunks.append(features.numpy().astype(np.float64))
        collected += features.shape[0]
    if not chunks:
        raise RuntimeError(f"No internal features found under {root}")
    return regularized_covariance(np.concatenate(chunks, axis=0), shrinkage, eps)


def matrix_sqrt(covariance: np.ndarray, inverse: bool, eps: float) -> torch.Tensor:
    matrix = torch.from_numpy(covariance).double()
    values, vectors = torch.linalg.eigh(matrix)
    values = torch.clamp(values, min=eps)
    powers = -0.5 if inverse else 0.5
    return vectors @ torch.diag(values.pow(powers)) @ vectors.t()


def align(features: torch.Tensor, source_stats, target_stats, eps: float) -> torch.Tensor:
    source_mean, source_cov = source_stats
    target_mean, target_cov = target_stats
    whitening = matrix_sqrt(source_cov, inverse=True, eps=eps)
    coloring = matrix_sqrt(target_cov, inverse=False, eps=eps)
    transform = whitening @ coloring
    source = features.double() - torch.from_numpy(source_mean).double()
    target = torch.from_numpy(target_mean).double()
    return (source @ transform + target).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--internal-csv", type=Path, default=Path("data/labels/internal_train.csv"))
    parser.add_argument("--external-csv", type=Path, default=Path("data/labels/external_test.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-internal-patches", type=int, default=1_000_000)
    parser.add_argument("--internal-sample-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--shrinkage", type=float, default=0.10)
    parser.add_argument("--eps", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_stats = collect_internal_stats(
        args.internal_csv,
        args.feature_root,
        args.max_internal_patches,
        args.internal_sample_rate,
        args.seed,
        args.shrinkage,
        args.eps,
    )

    table = pd.read_csv(args.external_csv)
    success = 0
    for _, row in tqdm(table.iterrows(), total=len(table), desc="CORAL external slides"):
        slide_id = str(row["slide_id"])
        case_id = str(row["case_id"]) if "case_id" in row and not pd.isna(row["case_id"]) else None
        feature_path, source_folder = find_feature(args.feature_root, slide_id, case_id)
        if feature_path is None:
            continue
        features, payload = load_features(feature_path)
        source_stats = regularized_covariance(features.numpy().astype(np.float64), args.shrinkage, args.eps)
        aligned = torch.clamp(align(features, source_stats, target_stats, args.eps), min=-10.0, max=10.0)

        destination = args.output_dir / (source_folder.name if source_folder and source_folder != args.feature_root else slide_id)
        destination.mkdir(parents=True, exist_ok=True)
        if payload is not None:
            payload["features"] = aligned
            torch.save(payload, destination / "image_feature.pt")
        else:
            torch.save(aligned, destination / "image_feature.pt")
        if source_folder and source_folder.is_dir():
            for metadata in source_folder.glob("*.json"):
                shutil.copy2(metadata, destination / metadata.name)
        (destination / "coral_metadata.json").write_text(
            json.dumps({
                "method": "CORAL",
                "source_statistics": "per_external_slide",
                "target_statistics": "internal_training_cohort",
                "uses_external_labels": False,
                "seed": args.seed,
                "shrinkage": args.shrinkage,
                "eps": args.eps,
            }, indent=2),
            encoding="utf-8",
        )
        success += 1
    print(f"CORAL completed: {success}/{len(table)} external slides written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
