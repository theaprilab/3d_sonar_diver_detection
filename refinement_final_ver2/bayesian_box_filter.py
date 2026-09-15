"""Causal, non-learning Bayesian/Kalman refinement for box dimensions and SO(3).

Dimension and rotation refinement are independent switches so their effects can
be ablated.  Detector score is never changed.  A constant-velocity Kalman state
is maintained for association; its posterior center can be emitted only through
the explicit ``refine_center`` option.

Dimension state (in log space):
    s_t^- = s_{t-1},  P_t^- = P_{t-1}
    z_t   = log(detection_dims) - train_measurement_bias
    R_t   = R_ref * n_ref / max(n_t, 1)
    K_t   = P_t^- (P_t^- + R_t)^-1
    s_t   = s_t^- + K_t (z_t - s_t^-)

All statistics are estimated once from the training split.  No validation-set
grid search and no learned refinement weights are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DIM_KEYS = ("l", "w", "h")
RANGE_BINS = (
    (0.0, 2.0, "0-2m"),
    (2.0, 3.5, "2-3.5m"),
    (3.5, 5.0, "3.5-5m"),
    (5.0, float("inf"), "5m+"),
)


def range_bucket(x: float, y: float) -> str:
    radius = float(np.hypot(x, y))
    for lo, hi, name in RANGE_BINS:
        if lo <= radius < hi:
            return name
    return "5m+"


def _as_cov(value: Iterable[Iterable[float]], floor: float = 1e-6) -> np.ndarray:
    """Return a finite symmetric positive-definite covariance."""
    cov = np.asarray(value, dtype=float)
    cov = 0.5 * (cov + cov.T)
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains a non-finite value")
    eig_min = float(np.linalg.eigvalsh(cov).min())
    if eig_min < floor:
        cov = cov + np.eye(cov.shape[0]) * (floor - eig_min)
    return cov


_RZ_PI = np.diag([-1.0, -1.0, 1.0])


def project_so3(matrix: np.ndarray) -> np.ndarray:
    """Project a noisy 3x3 matrix to the closest proper rotation."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, float))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def so3_exp(vector: np.ndarray) -> np.ndarray:
    """SO(3) exponential map from a tangent rotation vector."""
    vector = np.asarray(vector, float)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-10:
        return project_so3(np.eye(3) + np.array(
            [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]],
             [-vector[1], vector[0], 0.0]]))
    axis = vector / theta
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Numerically robust SO(3) logarithm as a tangent rotation vector."""
    rotation = project_so3(rotation)
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    vee = np.array([rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1]])
    if theta < 1e-8:
        return 0.5 * vee
    if np.pi - theta < 1e-5:
        values, vectors = np.linalg.eig(rotation)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        if np.dot(axis, vee) < 0:
            axis = -axis
        return theta * axis
    return theta * vee / (2.0 * np.sin(theta))


def rotation_innovation(reference: np.ndarray, measurement: np.ndarray,
                        fold_z_pi: bool = True) -> np.ndarray:
    """Right-invariant residual, optionally respecting the evaluator's z-pi box symmetry."""
    reference = project_so3(reference)
    measurement = project_so3(measurement)
    candidates = [so3_log(reference.T @ measurement)]
    if fold_z_pi:
        candidates.append(so3_log(reference.T @ (measurement @ _RZ_PI)))
    return min(candidates, key=lambda value: float(value @ value))


@dataclass(frozen=True)
class FilterConfig:
    """A-priori algorithm choices; none are selected on validation AP."""

    association_probability: float = 0.99
    max_missed_frames: int = 3
    conservative_oversize_only: bool = True
    refine_center: bool = False
    refine_dimensions: bool = True
    refine_rotation: bool = False
    fold_rotation_z_pi: bool = True
    emit_debug_fields: bool = False

    @property
    def association_chi2(self) -> float:
        # Exact 99% quantile of chi-square with two degrees of freedom.
        if not np.isclose(self.association_probability, 0.99):
            raise ValueError("only the pre-declared 0.99 association probability is supported")
        return 9.21034037197618


class TrainStatistics:
    """Train-only sufficient statistics consumed by the recursive filter."""

    def __init__(self, payload: dict):
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported statistics schema")
        self.payload = payload
        self.n_ref = max(float(payload["density_reference_median"]), 1.0)
        measurement = payload["measurement"]
        self.dim_bias = np.asarray(measurement["log_dim_bias"], dtype=float)
        self.dim_cov = _as_cov(measurement["log_dim_cov"])
        self.center_bias = np.asarray(measurement["center_bias_xy"], dtype=float)
        self.center_cov = _as_cov(measurement["center_cov_xy"])
        self.process_cov = _as_cov(payload["motion"]["displacement_cov_xy"])
        self.rotation_cov = _as_cov(measurement.get("rotation_cov_tangent", np.eye(3) * 0.25))
        self.rotation_process_cov = _as_cov(
            payload["motion"].get("rotation_process_cov_tangent", np.eye(3) * 0.01))
        self.priors = payload["dimension_prior_by_range"]
        self.global_prior = payload["dimension_prior_global"]

    @classmethod
    def load(cls, path: str | Path) -> "TrainStatistics":
        with open(path, encoding="utf-8") as handle:
            return cls(json.load(handle))

    def prior(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        item = self.priors.get(range_bucket(x, y), self.global_prior)
        if int(item.get("count", 0)) < 2:
            item = self.global_prior
        return np.asarray(item["mean_log_dims"], float), _as_cov(item["cov_log_dims"])

    def density_scale(self, n_points: float) -> float:
        # R(n)=R_ref*n_ref/n.  No clipping constant is tuned on validation data.
        return self.n_ref / max(float(n_points), 1.0)

    def dimension_measurement_cov(self, n_points: float) -> np.ndarray:
        return _as_cov(self.dim_cov * self.density_scale(n_points))

    def center_measurement_cov(self, n_points: float) -> np.ndarray:
        return _as_cov(self.center_cov * self.density_scale(n_points))

    def rotation_measurement_cov(self, n_points: float) -> np.ndarray:
        return _as_cov(self.rotation_cov * self.density_scale(n_points))


class _Track:
    def __init__(self, track_id: int, detection: dict, stats: TrainStatistics,
                 fold_rotation_z_pi: bool = True):
        self.track_id = track_id
        self.hits = 0
        self.missed = 0

        xy = np.array([detection["x"], detection["y"]], float) - stats.center_bias
        r_xy = stats.center_measurement_cov(detection.get("n", 0))
        self.center_mean = np.r_[xy, [0.0, 0.0]]
        self.center_cov = np.zeros((4, 4), float)
        self.center_cov[:2, :2] = r_xy
        self.center_cov[2:, 2:] = stats.process_cov * 4.0

        prior_mean, prior_cov = stats.prior(detection["x"], detection["y"])
        self.dim_mean = prior_mean
        self.dim_cov = prior_cov
        self.rotation_mean = project_so3(detection["R"])
        self.rotation_cov = stats.rotation_measurement_cov(detection.get("n", 0))
        self.update(detection, stats, initialize_center=False, initialize_rotation=False,
                    fold_rotation_z_pi=fold_rotation_z_pi)

    def predict(self, stats: TrainStatistics, dt: float = 1.0) -> None:
        f = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        q = stats.process_cov
        q_block = np.block(
            [[0.25 * dt**4 * q, 0.5 * dt**3 * q],
             [0.5 * dt**3 * q, dt**2 * q]]
        )
        self.center_mean = f @ self.center_mean
        self.center_cov = _as_cov(f @ self.center_cov @ f.T + q_block)
        # Physical dimensions are treated as static: Q_dim = 0.
        self.rotation_cov = _as_cov(self.rotation_cov + stats.rotation_process_cov * max(dt, 1.0))
        self.missed += 1

    def association_distance(self, detection: dict, stats: TrainStatistics) -> float:
        z = np.array([detection["x"], detection["y"]], float) - stats.center_bias
        innovation = z - self.center_mean[:2]
        s = _as_cov(self.center_cov[:2, :2] + stats.center_measurement_cov(detection.get("n", 0)))
        return float(innovation @ np.linalg.solve(s, innovation))

    def update(self, detection: dict, stats: TrainStatistics, initialize_center: bool = True,
               initialize_rotation: bool = True,
               fold_rotation_z_pi: bool = True) -> None:
        n_points = detection.get("n", 0)
        if initialize_center:
            z_xy = np.array([detection["x"], detection["y"]], float) - stats.center_bias
            h = np.zeros((2, 4), float)
            h[:, :2] = np.eye(2)
            r_xy = stats.center_measurement_cov(n_points)
            s_xy = _as_cov(h @ self.center_cov @ h.T + r_xy)
            k_xy = self.center_cov @ h.T @ np.linalg.inv(s_xy)
            innovation = z_xy - h @ self.center_mean
            self.center_mean = self.center_mean + k_xy @ innovation
            # Joseph form preserves covariance symmetry and positive semidefiniteness.
            identity = np.eye(4)
            ikh = identity - k_xy @ h
            self.center_cov = _as_cov(ikh @ self.center_cov @ ikh.T + k_xy @ r_xy @ k_xy.T)

        raw_dims = np.array([detection[key] for key in DIM_KEYS], float)
        if np.any(raw_dims <= 0):
            raise ValueError(f"non-positive dimensions in detection: {raw_dims}")
        z_dim = np.log(raw_dims) - stats.dim_bias
        r_dim = stats.dimension_measurement_cov(n_points)
        s_dim = _as_cov(self.dim_cov + r_dim)
        k_dim = self.dim_cov @ np.linalg.inv(s_dim)
        self.dim_mean = self.dim_mean + k_dim @ (z_dim - self.dim_mean)
        identity = np.eye(3)
        ik = identity - k_dim
        self.dim_cov = _as_cov(ik @ self.dim_cov @ ik.T + k_dim @ r_dim @ k_dim.T)

        if initialize_rotation:
            measurement_rotation = project_so3(detection["R"])
            innovation_rotation = rotation_innovation(
                self.rotation_mean, measurement_rotation, fold_z_pi=fold_rotation_z_pi)
            r_rotation = stats.rotation_measurement_cov(n_points)
            s_rotation = _as_cov(self.rotation_cov + r_rotation)
            k_rotation = self.rotation_cov @ np.linalg.inv(s_rotation)
            self.rotation_mean = project_so3(
                self.rotation_mean @ so3_exp(k_rotation @ innovation_rotation))
            ik_rotation = np.eye(3) - k_rotation
            self.rotation_cov = _as_cov(
                ik_rotation @ self.rotation_cov @ ik_rotation.T
                + k_rotation @ r_rotation @ k_rotation.T)
        self.hits += 1
        self.missed = 0

    def apply(self, detection: dict, config: FilterConfig) -> dict:
        result = dict(detection)
        raw_dims = np.array([detection[key] for key in DIM_KEYS], float)
        posterior_dims = np.exp(self.dim_mean)
        if config.conservative_oversize_only:
            posterior_dims = np.minimum(raw_dims, posterior_dims)
        if config.refine_dimensions:
            for key, value in zip(DIM_KEYS, posterior_dims):
                result[key] = float(value)
        if config.refine_rotation:
            result["R"] = self.rotation_mean.copy()
        if config.refine_center:
            result["x"], result["y"] = map(float, self.center_mean[:2])
        if config.emit_debug_fields:
            result["refinement_track_id"] = self.track_id
            result["refinement_hits"] = self.hits
            result["refinement_dim_std"] = np.sqrt(np.diag(self.dim_cov)).tolist()
            result["refinement_rotation_std_deg"] = np.degrees(
                np.sqrt(np.diag(self.rotation_cov))).tolist()
        return result


class BayesianBoxRefiner:
    """Scene-local causal tracker and Bayesian dimension estimator."""

    def __init__(self, statistics: TrainStatistics, config: FilterConfig | None = None):
        self.stats = statistics
        self.config = config or FilterConfig()
        self._tracks: list[_Track] = []
        self._next_track_id = 0

    def reset(self) -> None:
        self._tracks = []
        self._next_track_id = 0

    def process_frame(self, detections: list[dict], dt: float = 1.0) -> list[dict]:
        for track in self._tracks:
            track.predict(self.stats, dt=dt)

        pairs = []
        for detection_index, detection in enumerate(detections):
            for track_index, track in enumerate(self._tracks):
                distance = track.association_distance(detection, self.stats)
                if distance <= self.config.association_chi2:
                    pairs.append((distance, detection_index, track_index))
        pairs.sort(key=lambda item: item[0])

        assignments: dict[int, int] = {}
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        for _, detection_index, track_index in pairs:
            if detection_index in used_detections or track_index in used_tracks:
                continue
            assignments[detection_index] = track_index
            used_detections.add(detection_index)
            used_tracks.add(track_index)

        output = []
        for detection_index, detection in enumerate(detections):
            if detection_index in assignments:
                track = self._tracks[assignments[detection_index]]
                track.update(detection, self.stats,
                             fold_rotation_z_pi=self.config.fold_rotation_z_pi)
            else:
                track = _Track(self._next_track_id, detection, self.stats,
                               fold_rotation_z_pi=self.config.fold_rotation_z_pi)
                self._next_track_id += 1
                self._tracks.append(track)
            output.append(track.apply(detection, self.config))

        self._tracks = [
            track for track in self._tracks
            if track.missed <= self.config.max_missed_frames
        ]
        return output

    def process_scene(self, frames: list[list[dict]], frame_ids: list[int] | None = None) -> list[list[dict]]:
        self.reset()
        output = []
        previous_id = None
        for index, detections in enumerate(frames):
            frame_id = frame_ids[index] if frame_ids is not None else index
            dt = 1.0 if previous_id is None else max(float(frame_id - previous_id), 1.0)
            output.append(self.process_frame([dict(box) for box in detections], dt=dt))
            previous_id = frame_id
        return output
