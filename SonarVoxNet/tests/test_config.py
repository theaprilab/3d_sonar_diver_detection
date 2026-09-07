from pathlib import Path

from sonarvoxnet.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_main_recipe_is_the_sparse_6d_l1_model():
    config = load_config(ROOT / "configs/main_spconv_6d_l1.yaml")
    assert config["model"] == {"middle_encoder": "sparse", "detection_head": "center", "rotation_representation": "6d"}
    assert config["loss"]["rotation"] == "l1_target"


def test_ablation_inherits_the_fixed_recipe():
    config = load_config(ROOT / "configs/ablations/dense_middle_6d_l1.yaml")
    assert config["model"]["middle_encoder"] == "dense"
    assert config["training"]["optimizer"] == "adamw"
