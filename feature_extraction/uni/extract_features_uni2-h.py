#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNI2-h patch feature extraction for WSI MIL experiments.

Key revisions compared with the original script:
1. Fixes Config.SLIDE_DIRS vs Config.SLIDE_DIR compatibility.
2. Uses timm data config / create_transform when available, with a safe fallback.
3. Keeps patch_metadata.json as a list, consistent with MUSK and CONCH outputs.
4. Saves feature_extractor_metadata.json with feature dimension and slide-level metadata.
5. Checks feature/metadata length consistency and enforces the expected 1536-D UNI2-h output.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
from packaging import version
from torchvision import transforms
from tqdm import tqdm

from config import Config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.stain import MacenkoStainNormalizer

LOCAL_TIMM_PATH = getattr(Config, "LOCAL_TIMM_PATH", None)
if LOCAL_TIMM_PATH and os.path.isdir(LOCAL_TIMM_PATH):
    sys.path.insert(0, LOCAL_TIMM_PATH)

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

REQUIRED_TIMM_VERSION = "1.0.3"


class FeatureExtractor:
    def __init__(self):
        self.device = Config.DEVICE
        self.batch_size = Config.BATCH_SIZE_EXTRACT_FEATURE
        self.max_workers = Config.MAX_NUM_WORKERS
        self.max_tiles_per_slide = Config.MAX_TILES_PER_SLIDE
        self.local_weight_path = Config.UNI2_WEIGHT_PATH
        self.feature_dir = Config.FEATURE_DIR
        self.expected_feature_dim = int(getattr(Config, "EXPECTED_FEATURE_DIM", 1536))
        self.detected_feature_dim: int | None = None
        self.stain_normalizer = MacenkoStainNormalizer()

        self._set_seed(Config.SEED)
        self._check_environment()
        self.model, self.transform = self._load_uni2_h_model_locally()

        os.makedirs(self.feature_dir, exist_ok=True)
        print("=" * 70)
        print("UNI2-h feature extractor initialized")
        print(f"Device: {self.device}")
        print(f"timm version: {timm.__version__}")
        print(f"Weights: {self.local_weight_path}")
        print(f"Feature dir: {self.feature_dir}")
        print(f"Batch size: {self.batch_size}")
        print(f"Workers: {self.max_workers}")
        print(f"MAX_TILES_PER_SLIDE: {self.max_tiles_per_slide}")
        print(f"Expected feature dim: {self.expected_feature_dim}")
        print("=" * 70)

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _get_slide_dirs() -> List[str]:
        if hasattr(Config, "SLIDE_DIRS"):
            dirs = getattr(Config, "SLIDE_DIRS")
            if isinstance(dirs, str):
                return [dirs]
            return list(dirs)
        if hasattr(Config, "SLIDE_DIR"):
            return [getattr(Config, "SLIDE_DIR")]
        raise AttributeError("Config must define SLIDE_DIR or SLIDE_DIRS")

    def _check_environment(self):
        if not os.path.exists(self.local_weight_path):
            raise FileNotFoundError(f"UNI2-h local weight not found: {self.local_weight_path}")
        if version.parse(timm.__version__) < version.parse(REQUIRED_TIMM_VERSION):
            raise RuntimeError(
                f"Current timm={timm.__version__} is too old for the official UNI2-h configuration. "
                f"Please use timm >= {REQUIRED_TIMM_VERSION}."
            )

    @staticmethod
    def _safe_torch_load(path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_uni2_h_model_locally(self):
        print("Creating UNI2-h model with the official architecture...")
        timm_kwargs = {
            "model_name": "vit_giant_patch14_224",
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
            "pretrained": False,
        }
        model = timm.create_model(**timm_kwargs)

        print(f"Loading local weights: {self.local_weight_path}")
        state_dict = self._safe_torch_load(self.local_weight_path)
        if isinstance(state_dict, dict):
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        msg = model.load_state_dict(state_dict, strict=True)
        print(f"Weight loading report: {msg}")

        try:
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            print("Using timm resolved UNI2-h transform.")
        except Exception as e:
            logging.warning("Could not resolve timm transform; using safe fallback. Error: %s", e)
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        print("UNI2-h model loaded.")
        return model, transform

    def extract_from_directory(self):
        valid_slide_ids = self._collect_valid_slide_ids()
        if not valid_slide_ids:
            print("No valid slides found.")
            return

        overwrite = getattr(Config, "FEATURE_EXTRACTION_MODE", 1) == 2
        unprocessed_ids = []
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

        main_bar = tqdm(unprocessed_ids, desc="Extracting UNI2-h features", unit="slide", dynamic_ncols=True)
        success_count = 0
        failed_count = 0

        for slide_id in main_bar:
            try:
                start_time = time.time()
                feats, patch_meta, extractor_meta = self.extract_patch_features(slide_id)

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
        candidate_slide_ids: List[str]
        if os.path.exists(Config.LABEL_FILE):
            try:
                if Config.LABEL_FILE.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(Config.LABEL_FILE)
                else:
                    df = pd.read_csv(Config.LABEL_FILE)
                if "slide_id" in df.columns:
                    candidate_slide_ids = sorted(df["slide_id"].astype(str).unique().tolist())
                else:
                    candidate_slide_ids = self._scan_slide_ids_from_dirs()
            except Exception:
                candidate_slide_ids = self._scan_slide_ids_from_dirs()
        else:
            candidate_slide_ids = self._scan_slide_ids_from_dirs()

        valid_slide_ids = []
        for slide_id in candidate_slide_ids:
            h5_path = os.path.join(Config.PATCH_DIR, f"{slide_id}{Config.H5_EXT}")
            slide_path = self._find_slide_path(slide_id)
            if os.path.exists(h5_path) and slide_path is not None:
                valid_slide_ids.append(slide_id)
        return valid_slide_ids

    def _scan_slide_ids_from_dirs(self) -> List[str]:
        slide_ids = set()
        for slide_dir in self._get_slide_dirs():
            if not os.path.isdir(slide_dir):
                continue
            for fname in os.listdir(slide_dir):
                for ext in Config.SUPPORTED_SLIDE_EXTS:
                    if fname.lower().endswith(ext.lower()):
                        slide_ids.add(fname[: -len(ext)])
                        break
        return sorted(slide_ids)

    def _find_slide_path(self, slide_id: str) -> Optional[str]:
        for slide_dir in self._get_slide_dirs():
            if not os.path.isdir(slide_dir):
                continue
            for ext in Config.SUPPORTED_SLIDE_EXTS:
                candidate = os.path.join(slide_dir, f"{slide_id}{ext}")
                if os.path.exists(candidate):
                    return candidate
        return None

    @staticmethod
    def _read_h5(h5_path: str) -> Tuple[np.ndarray, int, int, float]:
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            attrs = f["coords"].attrs
            patch_level = int(attrs.get("patch_level", 0))
            patch_size = int(attrs.get("patch_size", 512))
            custom_downsample = float(attrs.get("custom_downsample", 1.0))
        return coords, patch_level, patch_size, custom_downsample

    def extract_patch_features(self, slide_id: str):
        h5_path = os.path.join(Config.PATCH_DIR, f"{slide_id}{Config.H5_EXT}")
        slide_path = self._find_slide_path(slide_id)
        if slide_path is None:
            raise FileNotFoundError(f"Slide not found: {slide_id}")

        coords, patch_level, patch_size, custom_downsample = self._read_h5(h5_path)
        if coords is None or len(coords) == 0:
            raise ValueError(f"Empty coords in {h5_path}")

        original_n = len(coords)
        if self.max_tiles_per_slide is not None and len(coords) > self.max_tiles_per_slide:
            keep_idx = np.linspace(0, len(coords) - 1, num=self.max_tiles_per_slide, dtype=int)
            coords = coords[keep_idx]

        base_mpp = None
        effective_mpp = None
        level_dimensions = None
        slide = openslide.OpenSlide(slide_path)
        try:
            try:
                base_mpp_raw = slide.properties.get(openslide.PROPERTY_NAME_MPP_X, None)
                base_mpp = float(base_mpp_raw) if base_mpp_raw is not None else None
            except Exception:
                base_mpp = None
            level_downsample = float(slide.level_downsamples[patch_level])
            effective_mpp = base_mpp * level_downsample if base_mpp is not None else None
            level_dimensions = [list(map(int, d)) for d in slide.level_dimensions]
        finally:
            slide.close()

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
            features = torch.empty((0, self.expected_feature_dim), dtype=torch.float32)

        if features.shape[0] != len(patch_meta):
            raise RuntimeError(f"Feature/metadata mismatch for {slide_id}: {features.shape[0]} vs {len(patch_meta)}")

        extractor_meta = {
            "slide_id": slide_id,
            "slide_path": slide_path,
            "h5_path": h5_path,
            "feature_extractor": "UNI2-h",
            "architecture": "vit_giant_patch14_224",
            "n_patches_total": int(original_n),
            "n_patches_used": int(features.shape[0]),
            "feature_dim": int(features.shape[1]) if features.ndim == 2 else self.expected_feature_dim,
            "stain_normalization": "Macenko",
            "patch_level": int(patch_level),
            "patch_size": int(patch_size),
            "custom_downsample": float(custom_downsample),
            "base_mpp": base_mpp,
            "effective_mpp": effective_mpp,
            "level_dimensions": level_dimensions,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return features, patch_meta, extractor_meta

    def _encode_with_oom_fallback(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = x.to(self.device, non_blocking=True)
            with torch.inference_mode():
                feat = self.model(x)
            return feat.cpu()
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
                    sub_feats.append(self.model(xb).cpu())
            return torch.cat(sub_feats, dim=0)

    def _check_feature_batch(self, feats: torch.Tensor):
        if feats.ndim != 2:
            feats = feats.reshape(feats.shape[0], -1)
        dim = int(feats.shape[1])
        if self.detected_feature_dim is None:
            self.detected_feature_dim = dim
            print(f"Detected UNI2-h feature dim: {dim}")
        elif dim != self.detected_feature_dim:
            raise ValueError(f"Inconsistent UNI2-h feature dim: {dim} vs {self.detected_feature_dim}")
        if dim != self.expected_feature_dim:
            raise ValueError(f"UNI2-h feature dim {dim} != EXPECTED_FEATURE_DIM {self.expected_feature_dim}")

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
    print("===== UNI2-h image feature extraction =====")
    extractor = FeatureExtractor()
    extractor.extract_from_directory()


if __name__ == "__main__":
    main()
