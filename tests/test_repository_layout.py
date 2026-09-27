"""Fast checks for a clean public source release."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_entry_points_exist():
    expected = [
        "scripts/train_main.py",
        "scripts/test_main.py",
        "scripts/train_baselines.py",
        "scripts/test_baselines.py",
        "scripts/train_ablation.py",
        "scripts/test_ablation.py",
        "scripts/make_heatmaps.py",
    ]
    assert all((ROOT / item).is_file() for item in expected)


def test_public_configs_exist():
    expected = [
        "configs/MambaMIL_DynamicWeight.yaml",
        "configs/MambaMIL.yaml",
        "configs/TransMIL.yaml",
        "configs/biospy_case_abmil.yaml",
        "configs/MambaMIL_Encoder_ResNet50.yaml",
        "configs/MambaMIL_Encoder_CONCH.yaml",
        "configs/MambaMIL_Encoder_UNI2-h.yaml",
    ]
    assert all((ROOT / item).is_file() for item in expected)


def test_public_extractor_contract_exists():
    expected = [
        "feature_extraction/resnet50/extract_features_resnet50.py",
        "feature_extraction/musk/extract_features_musk.py",
        "feature_extraction/common/stain.py",
        "feature_extraction/uni/extract_features_uni2-h.py",
        "feature_extraction/conch/extract_features_conch.py",
    ]
    assert all((ROOT / item).is_file() for item in expected)


def test_public_defaults_use_seed_421():
    for path in (ROOT / "configs").glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "seed: 421" in text, path


def test_no_private_path_markers_in_source():
    markers = ("/nas_3", "/server03_data", "newnas_1", "api_key=\"", "65ej")
    candidates = list((ROOT / "scripts").rglob("*.py"))
    candidates += list((ROOT / "configs").rglob("*.yaml"))
    candidates += list((ROOT / "feature_extraction").rglob("*.py"))
    for path in candidates:
        if path.name == "validate_release.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in markers), path
