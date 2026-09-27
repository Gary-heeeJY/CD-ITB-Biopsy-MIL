"""Shared Macenko stain normalization for all feature extractors.

The manuscript applies Macenko normalization before every frozen encoder.  A
normalizer is fitted lazily on the first valid tissue patch and then reused for
the remaining patches in the run.  This keeps the repository self-contained:
no private reference slide or hidden color template is required.  An optional
reference image can still be supplied when a laboratory has a fixed stain
template.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class MacenkoStainNormalizer:
    """Thread-safe Macenko normalizer with a deterministic fallback path."""

    def __init__(self, reference_path: str | Path | None = None):
        try:
            import torchstain
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Macenko normalization requires torchstain. Install the "
                "dependencies from requirements.txt."
            ) from exc

        self._normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        self._lock = threading.RLock()
        self._fitted = False
        self.reference_path = str(reference_path) if reference_path else None

        if reference_path:
            reference = Image.open(reference_path).convert("RGB")
            self.fit(reference)

    @staticmethod
    def _to_hwc_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        return torch.from_numpy(array)

    @staticmethod
    def _to_image(value: Any) -> Image.Image:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        value = np.clip(value, 0, 255).astype(np.uint8)
        return Image.fromarray(value, mode="RGB")

    def fit(self, image: Image.Image) -> None:
        """Fit the target stain statistics from one RGB image."""
        with self._lock:
            if self._fitted:
                return
            try:
                self._normalizer.fit(I=self._to_hwc_tensor(image))
                self._fitted = True
            except Exception:
                # Background or nearly empty patches can make Macenko fitting
                # singular; the next valid patch will be tried instead.
                self._fitted = False

    def normalize(self, image: Image.Image) -> Image.Image:
        """Return a normalized RGB image, preserving the input on failure."""
        original = image.convert("RGB")
        with self._lock:
            if not self._fitted:
                self.fit(original)
            if not self._fitted:
                return original
            try:
                normalized, _, _ = self._normalizer.normalize(
                    I=self._to_hwc_tensor(original), stains=True
                )
                return self._to_image(normalized)
            except Exception:
                return original

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.normalize(image)

    @property
    def fitted(self) -> bool:
        return self._fitted

