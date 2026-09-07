from pathlib import Path

from sonarvoxnet.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_main_recipe_is_the_sparse_6d_l1_model():
    config = load_config(ROOT / "configs/main_spconv_6d_l1.yaml")
    assert config["model"] == {"middle_encoder": "sparse", "detection_head": "center", "rotation_representation": "6d"}
    assert config["loss"]["rotation"] == "l1_target"


def test_main_recipe_uses_a_fixed_training_recipe():
    config = load_config(ROOT / "configs/main_spconv_6d_l1.yaml")
    assert config["training"]["optimizer"] == "adamw"


def test_head_backbone_recipes_change_only_their_named_condition():
    anchor = load_config(ROOT / "configs/ablations/anchor_zyaw.yaml")
    center = load_config(ROOT / "configs/ablations/center_zyaw.yaml")
    dense = load_config(ROOT / "configs/ablations/dense_middle_6d_l1.yaml")
    assert anchor["model"]["detection_head"] == "anchor"
    assert anchor["model"]["rotation_target"] == center["model"]["rotation_target"] == "zyaw"
    assert dense["model"]["middle_encoder"] == "dense"
    assert dense["model"]["detection_head"] == "center"
