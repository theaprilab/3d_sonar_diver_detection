"""Immutable method manifest selected on val_full LOSO before final fitting.

This module is metadata, not a second implementation of the filters.  Future
entrypoints should record ``FROZEN_GEOMETRY`` verbatim in their provenance.
"""

FROZEN_GEOMETRY_VERSION = "geometry-freeze-official-detector-v1"

FROZEN_GEOMETRY = {
    "version": FROZEN_GEOMETRY_VERSION,
    "detector": "voxelnet_otfsp_stdsafe_asp_emp_gtcurr65floor1_rngv2_rawbest_final_s{0,1}",
    "selection_split": "val_full_loso_oof",
    "evaluation": "canonical confidence-greedy one-to-one 3D-IoU TP",
    "order": (
        "neutral_score",
        "robust_bidirectional_size_measurement",
        "center_v2_gain_0.5_measurement",
        "fixed_sensor_stationary_cv_maneuver_center_imm",
        "point_conditioned_rotation_measurement",
        "point_conditioned_covariance_so3_imm",
        "final_nms",
    ),
    "center": {
        "learned_measurement": "center-roi-cap-safe-v2",
        "learned_gain": 0.5,
        "temporal_filter": "fixed-sensor stationary/CV/maneuver IMM",
        "adaptive_temporal_covariance": False,
    },
    "size": {
        "primary": "training-free robust bidirectional point support",
        "learned_late_size": False,
        "temporal_model": None,
    },
    "rotation": {
        "measurement": "epoch10 cosine-Tmax20 point-conditioned tangent residual",
        "covariance": "nested-LOSO point-conditioned diagonal tangent covariance",
        "temporal_filter": "causal stationary/CAV/maneuver SO(3) IMM",
    },
    "test_split_read_for_selection": False,
}

