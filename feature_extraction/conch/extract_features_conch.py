#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONCH patch feature extraction for WSI MIL experiments.

Key revisions compared with the original script:
1. Uses CONCH image embeddings recommended for MIL by default:
   encode_image(..., proj_contrast=False, normalize=False).
2. Keeps patch_metadata.json as a list, consistent with MUSK outputs.
3. Saves feature_extractor_metadata.json with feature dimension and encoder settings.
4. Checks feature/metadata length consistency and feature dimension consistency.
5. Skips failed patches rather than appending zero tensors that break metadata alignment.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import openslide
import torch
from tqdm import tqdm

from config import Config

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.stain import MacenkoStainNormalizer

# ====================== CONCH import ======================
# Keep this import path because your local project already uses it.
from conch_model.factory import create_model_from_pretrained


class FeatureExtractor:
    def __init__(self, batch_size: int | None = None, max_workers: int | None = None):
        self.device = Config.DEVICE
        self.batch_size = batch_size or Config.BATCH_SIZE_Extract_Feature
        self.max_workers = max_workers or Config.MAX_NUM_WORKERS
        self.feature_dir = Config.FEATURE_DIR
        self.expected_feature_dim = getattr(Config, "EXPECTED_FEATURE_DIM", None)
        self.detected_feature_dim: int | None = None

        self.model, self.transform = self._load_conch_model()

        self.stain_normalizer = MacenkoStainNormalizer()

        os.makedirs(self.feature_dir, exist_ok=True)
        print("=" * 70)
        print("CONCH feature extractor initialized")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Workers: {self.max_workers}")
        print(f"Feature dir: {self.feature_dir}")
        print(f"CONCH proj_contrast: {Config.CONCH_PROJ_CONTRAST}")
        print(f"CONCH normalize: {Config.CONCH_NORMALIZE}")
        print(f"Expected feature dim: {self.expected_feature_dim}")
        print("=" * 70)

    def _load_conch_model(self):
        """Load local CONCH model and official preprocessing transform."""
        print("Loading CONCH model...")
        model_cfg = getattr(Config, "CONCH_MODEL_CFG", "conch_vitb")

        # Your local factory uses keyword arguments. The fallback is for official-style APIs.
        try:
            model, transform = create_model_from_pretrained(
                model_cfg=model_cfg,
                checkpoint_path=Config.CONCH_MODEL_PATH,
                device=self.device,
            )
        except TypeError:
            model, transform = create_model_from_pretrained(model_cfg, Config.CONCH_MODEL_PATH)

        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad = False

        print("CONCH model loaded.")
        return model, transform

    def extract_from_directory(self):
        valid_slide_ids = self._collect_valid_slide_ids()
        if not valid_slide_ids:
            print("No valid slides found.")
            return

        unprocessed_ids = []
        overwrite = getattr(Config, "FEATURE_EXTRACTION_MODE", 1) == 2
        for slide_id in valid_slide_ids:
            save_dir = os.path.join(self.feature_dir, slide_id)
            feature_file = os.path.join(save_dir, "image_feature.pt")
            meta_file = os.path.join(save_dir, "patch_metadata.json")
            extractor_meta_file = os.path.join(save_dir, "feature_extractor_metadata.json")
            if not overwrite and os.path.exists(feature_file) and os.path.exists(meta_file) and os.path.exists(extractor_meta_file):
                continue
            unprocessed_ids.append(slide_id)

        if not unprocessed_ids:
            print("All slides have been processed. Nothing to do.")
            return

        main_bar = tqdm(unprocessed_ids, desc="Extracting CONCH features", unit="slide", dynamic_ncols=True)
        success_count = 0
        failed_count = 0

        for slide_id in main_bar:
            try:
                start_time = time.time()
                feats, patch_meta, extractor_meta = self.extract_patch_features_optimized(slide_id)

                save_dir = os.path.join(self.feature_dir, slide_id)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(feats, os.path.join(save_dir, "image_feature.pt"))

                with open(os.path.join(save_dir, "patch_metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(patch_meta, f, indent=2, ensure_ascii=False)

                with open(os.path.join(save_dir, "feature_extractor_metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(extractor_meta, f, indent=2, ensure_ascii=False)

                success_count += 1
                main_bar.set_postfix({
                    "success": success_count,
                    "failed": failed_count,
                    "time": f"{time.time() - start_time:.1f}s",
                    "shape": tuple(feats.shape),
                })
            except Exception as e:
                failed_count += 1
                main_bar.write(f"Failed {slide_id}: {str(e)}")
                logging.error("Feature extraction failed for slide_id=%s\n%s", slide_id, traceback.format_exc())

        print("=" * 70)
        print(f"Finished. Success: {success_count}/{len(unprocessed_ids)}, failed: {failed_count}")
        print("=" * 70)

    def _collect_valid_slide_ids(self) -> List[str]:
        slide_ext = Config.SLIDE_EXT
        slide_ids = []
        for fname in os.listdir(Config.SLIDE_DIR):
            if fname.lower().endswith(slide_ext.lower()):
                slide_ids.append(fname[: -len(slide_ext)])
        slide_ids = sorted(slide_ids)

        valid_slide_ids = []
        for slide_id in slide_ids:
            h5_path = os.path.join(Config.PATCH_DIR, f"{slide_id}{Config.H5_EXT}")
            slide_path = os.path.join(Config.SLIDE_DIR, f"{slide_id}{Config.SLIDE_EXT}")
            if os.path.exists(h5_path) and os.path.exists(slide_path):
                valid_slide_ids.append(slide_id)
            else:
                logging.warning("Missing h5 or slide for %s", slide_id)
        return valid_slide_ids

    @staticmethod
    def _read_h5(h5_path: str) -> Tuple[np.ndarray, int, int, float]:
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            attrs = f["coords"].attrs
            patch_level = int(attrs.get("patch_level", 0))
            patch_size = int(attrs.get("patch_size", 512))
            custom_downsample = float(attrs.get("custom_downsample", 1.0))
        return coords, patch_level, patch_size, custom_downsample

    def extract_patch_features_optimized(self, slide_id: str):
        h5_path = os.path.join(Config.PATCH_DIR, f"{slide_id}{Config.H5_EXT}")
        slide_path = os.path.join(Config.SLIDE_DIR, f"{slide_id}{Config.SLIDE_EXT}")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 not found: {h5_path}")
        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"Slide not found: {slide_path}")

        coords, patch_level, patch_size, custom_downsample = self._read_h5(h5_path)
        if coords is None or len(coords) == 0:
            raise ValueError(f"Empty coords in {h5_path}")

        original_n = len(coords)
        all_features: List[torch.Tensor] = []
        patch_meta: List[Dict] = []
        n_batches = (len(coords) + self.batch_size - 1) // self.batch_size

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            pbar = tqdm(range(n_batches), leave=False, desc=slide_id, dynamic_ncols=True)
            for i in pbar:
                s = i * self.batch_size
                e = min(s + self.batch_size, len(coords))
                batch_coords = coords[s:e]

                futures = [
                    pool.submit(self._process_one_patch, slide_path, c, patch_level, patch_size)
                    for c in batch_coords
                ]

                imgs = []
                valid_meta_this_batch = []
                for fut, c in zip(futures, batch_coords):
                    try:
                        imgs.append(fut.result())
                        valid_meta_this_batch.append({
                            "coord": [int(c[0]), int(c[1])],
                            "patch_level": int(patch_level),
                            "patch_size": int(patch_size),
                            "custom_downsample": float(custom_downsample),
                        })
                    except Exception:
                        logging.error("Patch reading failed for %s coord=%s\n%s", slide_id, c, traceback.format_exc())
                        continue

                if not imgs:
                    continue

                x = torch.stack(imgs, dim=0)
                try:
                    feats = self._encode_with_oom_fallback(x)
                    self._check_feature_batch(feats)
                    all_features.append(feats.cpu())
                    patch_meta.extend(valid_meta_this_batch)
                except Exception:
                    logging.error("Forward failed for %s batch=%d\n%s", slide_id, i, traceback.format_exc())
                    continue
            pbar.close()

        if all_features:
            features = torch.cat(all_features, dim=0).float()
        else:
            dim = self.expected_feature_dim or self.detected_feature_dim or 0
            features = torch.empty((0, dim), dtype=torch.float32)

        if features.shape[0] != len(patch_meta):
            raise RuntimeError(f"Feature/metadata mismatch for {slide_id}: {features.shape[0]} vs {len(patch_meta)}")

        feature_dim = int(features.shape[1]) if features.ndim == 2 and features.numel() > 0 else (self.expected_feature_dim or self.detected_feature_dim)
        extractor_meta = {
            "slide_id": slide_id,
            "slide_path": slide_path,
            "h5_path": h5_path,
            "feature_extractor": "CONCH",
            "model_cfg": getattr(Config, "CONCH_MODEL_CFG", "conch_vitb"),
            "checkpoint_path": Config.CONCH_MODEL_PATH,
            "proj_contrast": bool(Config.CONCH_PROJ_CONTRAST),
            "normalize": bool(Config.CONCH_NORMALIZE),
            "n_patches_total": int(original_n),
            "n_patches_used": int(features.shape[0]),
            "feature_dim": int(feature_dim) if feature_dim is not None else None,
            "stain_normalization": "Macenko",
            "patch_level": int(patch_level),
            "patch_size": int(patch_size),
            "custom_downsample": float(custom_downsample),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return features, patch_meta, extractor_meta

    def _encode_with_oom_fallback(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = x.to(self.device, non_blocking=True)
            with torch.inference_mode():
                return self._encode_batch(x).cpu()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            logging.error("CUDA OOM; retrying with smaller sub-batches")
            torch.cuda.empty_cache()
            sub_b = max(1, min(8, x.size(0) // 4 if x.size(0) > 4 else 1))
            sub_feats = []
            for j in range(0, x.size(0), sub_b):
                xb = x[j:j + sub_b].to(self.device, non_blocking=True)
                with torch.inference_mode():
                    sub_feats.append(self._encode_batch(xb).cpu())
            return torch.cat(sub_feats, dim=0)

    def _encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "encode_image"):
            feats = self.model.encode_image(
                x,
                proj_contrast=Config.CONCH_PROJ_CONTRAST,
                normalize=Config.CONCH_NORMALIZE,
            )
        else:
            feats = self.model(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim != 2:
            feats = feats.reshape(feats.shape[0], -1)
        return feats.float()

    def _check_feature_batch(self, feats: torch.Tensor):
        if feats.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {tuple(feats.shape)}")
        dim = int(feats.shape[1])
        if self.detected_feature_dim is None:
            self.detected_feature_dim = dim
            print(f"Detected CONCH feature dim: {dim}")
        elif dim != self.detected_feature_dim:
            raise ValueError(f"Inconsistent CONCH feature dim: {dim} vs {self.detected_feature_dim}")
        if self.expected_feature_dim is not None and dim != int(self.expected_feature_dim):
            raise ValueError(f"CONCH feature dim {dim} != EXPECTED_FEATURE_DIM {self.expected_feature_dim}")

    def _process_one_patch(self, slide_path: str, coord: Sequence[int], level: int, size: int) -> torch.Tensor:
        wsi = openslide.open_slide(slide_path)
        try:
            img = wsi.read_region((int(coord[0]), int(coord[1])), int(level), (int(size), int(size))).convert("RGB")
            img = self.stain_normalizer(img)
            return self.transform(img)
        finally:
            try:
                wsi.close()
            except Exception:
                pass


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print("===== CONCH image feature extraction =====")
    extractor = FeatureExtractor()
    extractor.extract_from_directory()


if __name__ == "__main__":
    main()
