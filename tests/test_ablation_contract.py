from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_case_variant_module_declares_all_four_variants():
    source = (ROOT / "framework/engine/case_variants.py").read_text(encoding="utf-8")
    for name in (
        "Case_predict_MeanPool",
        "Case_predict_AFSOnly",
        "Case_predict_TransOnly",
        "Case_predict_Full",
    ):
        assert f"class {name}" in source
    for variant in "ABCD":
        assert f'"{variant}":' in source or f"'{variant}':" in source


def test_ablation_entry_points_expose_attention_and_transformer_controls():
    train_source = (ROOT / "scripts/train_ablation.py").read_text(encoding="utf-8")
    test_source = (ROOT / "scripts/test_ablation.py").read_text(encoding="utf-8")
    for source in (train_source, test_source):
        assert "attention_only" in source
        assert "transformer_only" in source


def test_public_test_script_uses_validation_threshold_only():
    source = (ROOT / "framework/scripts/test_MambaMIL.py").read_text(encoding="utf-8")
    assert "load_pooled_validation_threshold" in source
    assert "roc_curve" not in source
    assert "Case_predict_AFSOnly" in source
    assert "Case_predict_TransOnly" in source


def test_engine_exports_only_modules_present_in_the_release():
    source = (ROOT / "framework/engine/__init__.py").read_text(encoding="utf-8")
    assert "train_val_case_Trans" not in source
    assert (ROOT / "framework/engine/train_val_case_mamba.py").is_file()
    assert (ROOT / "framework/engine/train_val_patient_mil.py").is_file()
