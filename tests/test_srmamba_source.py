import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = (
    ROOT / "framework" / "models" / "MambaMIL.py",
    ROOT / "framework" / "Heatmap" / "models" / "MambaMIL.py",
)
SRMAMBA_PATHS = (
    ROOT / "framework" / "models" / "srmamba.py",
    ROOT / "framework" / "Heatmap" / "models" / "srmamba.py",
)


class TestSRMambaSource(unittest.TestCase):
    def test_both_model_entrypoints_use_the_sr_mamba_mixer(self):
        for path in MODEL_PATHS:
            source = path.read_text(encoding="utf-8")
            self.assertIn("from .srmamba import SRMamba", source)
            self.assertIn("mixer_class = SRMamba", source)
            self.assertIn("mixer(h, rate=self.rate)", source)


    def test_both_sr_mamba_modules_implement_rate_based_reordering(self):
        implementation = SRMAMBA_PATHS[0].read_text(encoding="utf-8")
        self.assertIn("class TransposeTokenReEmbedding", implementation)
        self.assertIn("transpose_normal_padding", implementation)
        self.assertIn("transpose_remove_padding", implementation)
        self.assertIn("rate", implementation)
        self.assertIn("out_forward + out_reordered", implementation)
        for parameter_name in ("conv1d_b", "x_proj_b", "dt_proj_b", "A_b_log", "D_b"):
            self.assertIn(parameter_name, implementation)
        wrapper = SRMAMBA_PATHS[1].read_text(encoding="utf-8")
        self.assertIn("from framework.models.srmamba import SRMamba", wrapper)


    def test_touched_model_sources_are_english_only(self):
        cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
        for path in MODEL_PATHS + SRMAMBA_PATHS:
            self.assertIsNone(cjk.search(path.read_text(encoding="utf-8")), path)


if __name__ == "__main__":
    unittest.main()
