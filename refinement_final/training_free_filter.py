"""Strict training-free, causal Bernoulli/Kalman box refinement.

No train/validation statistics are consumed.  Every adaptive quantity is reset
at the beginning of a scene and estimated recursively from that scene.  The
four interventions are independent so a single detector dump can support a
factorial ablation:

* existence: Bernoulli track-existence evidence and confidence suppression;
* point size: conservative point-support measurement for systematic oversize;
* temporal size: recursive log-dimension fusion;
* rotation: uncertainty-gated SO(3) recursive fusion.

The center constant-velocity Kalman state is always used for association but is
emitted only by the explicit ``refine_center`` ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sqrt
from typing import Iterable

import numpy as np

from bayesian_box_filter import (_as_cov, project_so3, rotation_innovation,
                                 so3_exp)


CHI2_2_99 = 9.21034037197618
CHI2_2_95 = 5.991464547107979
CHI2_3_99 = 11.344866730144373
CHI2_3_95 = 7.814727903251179
EPS = 1e-9


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = exp(max(value, -60.0))
    return z / (1.0 + z)


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return float(np.log(value) - np.log1p(-value))


def _gaussian_density(innovation: np.ndarray, covariance: np.ndarray) -> float:
    covariance = _as_cov(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return EPS
    dim = innovation.shape[0]
    mahalanobis = float(innovation @ np.linalg.solve(covariance, innovation))
    log_value = -0.5 * (dim * np.log(2.0 * np.pi) + logdet + mahalanobis)
    return float(max(np.exp(np.clip(log_value, -60.0, 60.0)), EPS))


class RunningCovariance:
    """Scene-local Welford scatter with a declared physical/numerical floor."""

    def __init__(self, dimension: int, floor: np.ndarray):
        self.dimension = dimension
        self.floor = _as_cov(floor)
        self.count = 0
        self.mean = np.zeros(dimension, dtype=float)
        self.m2 = np.zeros((dimension, dimension), dtype=float)

    def update(self, value: Iterable[float]) -> None:
        value = np.asarray(value, dtype=float)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += np.outer(delta, value - self.mean)

    def covariance(self) -> np.ndarray:
        if self.count < 2:
            return self.floor.copy()
        sample = self.m2 / (self.count - 1)
        # A sensor/numerical floor is additive: repeated quantized observations
        # are not treated as infinitely exact even when their sample scatter is 0.
        return _as_cov(sample + self.floor)


@dataclass(frozen=True)
class TrainingFreeConfig:
    """Pre-declared algorithmic and sensor constants; none are fitted to a split."""

    refine_existence: bool = False
    refine_point_size: bool = False
    refine_temporal_size: bool = False
    refine_rotation: bool = False
    refine_center: bool = False
    emit_debug_fields: bool = False

    # v3-c switches.  Defaults preserve the completed v3-b experiment.
    neutral_logodds_score: bool = False
    recover_low_score: bool = False
    robust_temporal_size: bool = False
    bidirectional_size: bool = False
    center_3d: bool = False

    # From the frozen detector/sensor grid, not from residual fitting.
    voxel_size_xyz: tuple[float, float, float] = (0.1, 0.1, 0.5)
    center_cell_xy: tuple[float, float] = (0.2, 0.2)
    search_area_m2: float = 120.0  # [0,12] x [-5,5]

    association_chi2: float = CHI2_2_99
    rotation_chi2: float = CHI2_3_99
    max_missed_frames: int = 3
    survival_probability: float = 0.99
    birth_existence: float = 0.5  # non-informative Bernoulli prior
    min_support_voxels: int = 8
    support_quantile: float = 0.975
    fold_rotation_z_pi: bool = True
    canonical_score_threshold: float = 0.30
    recovery_chi2: float = CHI2_3_95
    recovery_confirmation_hits: int = 2


def _hungarian_dense(cost: np.ndarray) -> list[tuple[int, int]]:
    """Rectangular O(n^3) Hungarian assignment without an external dependency."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost matrix must be two-dimensional")
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []
    transposed = n_rows > n_cols
    work = cost.T if transposed else cost
    n, m = work.shape  # n <= m
    finite = work[np.isfinite(work)]
    large = (float(np.max(finite)) + 1.0) * (n + m + 1.0) if finite.size else 1e9
    work = np.where(np.isfinite(work), work, large)

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for row in range(1, n + 1):
        p[0] = row
        col0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[col0] = True
            row0 = p[col0]
            delta = np.inf
            col1 = 0
            for col in range(1, m + 1):
                if used[col]:
                    continue
                current = work[row0 - 1, col - 1] - u[row0] - v[col]
                if current < minv[col]:
                    minv[col] = current
                    way[col] = col0
                if minv[col] < delta:
                    delta = minv[col]
                    col1 = col
            for col in range(m + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break
        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    pairs = []
    for col in range(1, m + 1):
        if p[col] == 0:
            continue
        row_index, col_index = p[col] - 1, col - 1
        pairs.append((col_index, row_index) if transposed else (row_index, col_index))
    return pairs


def _local_axis_resolution(rotation: np.ndarray, voxel_size_xyz: Iterable[float]) -> np.ndarray:
    """World voxel widths projected onto the three local box axes."""
    widths = np.asarray(voxel_size_xyz, dtype=float)
    rotation = project_so3(rotation)
    return np.sqrt((rotation * rotation).T @ (widths * widths))


def point_support_dimensions(detection: dict, points: np.ndarray,
                             config: TrainingFreeConfig) -> tuple[np.ndarray, dict]:
    """Return a conservative, at-most-one-local-cell shrink measurement.

    The statistic uses occupied local voxels and a symmetric absolute extent.
    It never enlarges a detector box.  Requiring an empty outer band plus an
    occupied inner region avoids interpreting a globally empty sonar crop as
    evidence for a smaller object.
    """
    raw = np.array([detection[key] for key in ("l", "w", "h")], dtype=float)
    debug = {"effective_voxels": 0, "support_quality": 0.0,
             "supported_dims": raw.tolist()}
    if points is None or len(points) == 0 or np.any(raw <= 0):
        return raw, debug
    center = np.array([detection["x"], detection["y"], detection["z"]], dtype=float)
    rotation = project_so3(detection["R"])
    local = (np.asarray(points, dtype=float) - center) @ rotation
    half = raw * 0.5
    inside = np.all(np.abs(local) <= half, axis=1)
    local = local[inside]
    if len(local) == 0:
        return raw, debug

    axis_resolution = _local_axis_resolution(rotation, config.voxel_size_xyz)
    cells = np.floor(local / np.maximum(axis_resolution, 1e-6)).astype(np.int64)
    _, unique_indices = np.unique(cells, axis=0, return_index=True)
    occupied = local[np.sort(unique_indices)]
    n_effective = len(occupied)
    debug["effective_voxels"] = int(n_effective)
    debug["support_quality"] = float(1.0 - np.exp(-n_effective / config.min_support_voxels))
    if n_effective < config.min_support_voxels:
        return raw, debug

    supported = raw.copy()
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        cross_section = np.all(np.abs(occupied[:, other]) <= half[other], axis=1)
        values = np.abs(occupied[cross_section, axis])
        if len(values) < config.min_support_voxels:
            continue
        resolution = float(axis_resolution[axis])
        outer_start = max(float(half[axis] - resolution), 0.0)
        outer_count = int(np.count_nonzero(values >= outer_start))
        if outer_count:
            continue
        robust_half = float(np.quantile(values, config.support_quantile) + 0.5 * resolution)
        evidence_dimension = 2.0 * robust_half
        # One resolution per frame is the maximum permitted contraction.  Long
        # tracks may accumulate further evidence through temporal fusion.
        supported[axis] = min(raw[axis], max(raw[axis] - resolution, evidence_dimension))
    debug["supported_dims"] = supported.tolist()
    return supported, debug


class _SceneEvidence:
    """Conjugate scene-local nuisance estimates shared by tracks in one refiner."""

    def __init__(self, config: TrainingFreeConfig):
        self.config = config
        self.clutter_alpha = 0.5
        self.clutter_beta = config.search_area_m2
        self.detection_alpha = 0.5
        self.detection_beta = 0.5

    @property
    def clutter_density(self) -> float:
        return self.clutter_alpha / max(self.clutter_beta, EPS)

    @property
    def detection_probability(self) -> float:
        return self.detection_alpha / (self.detection_alpha + self.detection_beta)

    def observe_frame(self, unassigned_detections: int) -> None:
        # A new detection receives r=0.5, hence its expected clutter mass is 0.5.
        self.clutter_alpha += 0.5 * unassigned_detections
        self.clutter_beta += self.config.search_area_m2

    def observe_hit(self, existence_mass: float) -> None:
        self.detection_alpha += float(np.clip(existence_mass, 0.0, 1.0))

    def observe_miss(self, existence_mass: float) -> None:
        self.detection_beta += float(np.clip(existence_mass, 0.0, 1.0))


class _TrainingFreeTrack:
    def __init__(self, track_id: int, detection: dict, measurement_dims: np.ndarray,
                 config: TrainingFreeConfig):
        self.track_id = track_id
        self.hits = 1
        self.missed = 0
        self.existence = config.birth_existence

        center_dimension = 3 if config.center_3d else 2
        position = np.array(
            [detection[key] for key in (("x", "y", "z") if config.center_3d else ("x", "y"))],
            dtype=float)
        cell = np.asarray(
            (*config.center_cell_xy, config.voxel_size_xyz[2])
            if config.center_3d else config.center_cell_xy, dtype=float)
        self.center_dimension = center_dimension
        self.center_sensor_cov = np.diag(cell * cell / 12.0)
        self.center_measure_cov = self.center_sensor_cov.copy()
        self.center_process_cov = self.center_sensor_cov.copy()
        self.center_mean = np.r_[position, np.zeros(center_dimension)]
        self.center_cov = np.zeros((2 * center_dimension, 2 * center_dimension), dtype=float)
        self.center_cov[:center_dimension, :center_dimension] = self.center_measure_cov
        self.center_cov[center_dimension:, center_dimension:] = self.center_measure_cov * 4.0
        self.center_innovations = RunningCovariance(center_dimension, self.center_sensor_cov)
        self.velocity_steps = RunningCovariance(center_dimension, self.center_sensor_cov)

        self.dim_mean = np.log(measurement_dims)
        self.dim_cov = self._dimension_floor(detection, config)
        self.dim_process_floor = self.dim_cov.copy()
        self.dim_innovations = RunningCovariance(3, self.dim_cov)

        self.rotation_mean = project_so3(detection["R"])
        initial_rotation_cov = np.eye(3) * (pi / 4.0) ** 2
        rotation_floor = np.eye(3) * np.radians(0.1) ** 2
        self.rotation_cov = initial_rotation_cov
        self.rotation_innovations = RunningCovariance(3, rotation_floor)
        self.rotation_steps = RunningCovariance(3, rotation_floor)
        self.previous_measured_rotation = self.rotation_mean.copy()

    @staticmethod
    def _dimension_floor(detection: dict, config: TrainingFreeConfig) -> np.ndarray:
        dims = np.array([detection[key] for key in ("l", "w", "h")], dtype=float)
        resolution = _local_axis_resolution(detection["R"], config.voxel_size_xyz)
        sigma_log = np.maximum(resolution / sqrt(12.0) / np.maximum(dims, 1e-3), 1e-3)
        return np.diag(sigma_log * sigma_log)

    def predict(self, dt: float, evidence: _SceneEvidence, config: TrainingFreeConfig) -> None:
        dimension = self.center_dimension
        transition = np.block([
            [np.eye(dimension), dt * np.eye(dimension)],
            [np.zeros((dimension, dimension)), np.eye(dimension)],
        ])
        q = self.center_process_cov
        q_block = np.block([[0.25 * dt**4 * q, 0.5 * dt**3 * q],
                            [0.5 * dt**3 * q, dt**2 * q]])
        self.center_mean = transition @ self.center_mean
        self.center_cov = _as_cov(transition @ self.center_cov @ transition.T + q_block)
        if config.robust_temporal_size:
            # A sensor-derived random-walk floor prevents a long track from
            # becoming irreversibly certain after a poor early observation.
            self.dim_cov = _as_cov(
                self.dim_cov + self.dim_process_floor * max(float(dt), 1.0))
        self.rotation_cov = _as_cov(
            self.rotation_cov + self.rotation_steps.covariance() * max(dt, 1.0))
        self.existence = float(np.clip(
            config.survival_probability ** max(dt, 1.0) * self.existence, EPS, 1.0 - EPS))
        self.missed += 1

    def association(self, detection: dict) -> tuple[float, np.ndarray, np.ndarray]:
        keys = ("x", "y", "z") if self.center_dimension == 3 else ("x", "y")
        measurement = np.array([detection[key] for key in keys], dtype=float)
        innovation = measurement - self.center_mean[:self.center_dimension]
        covariance = _as_cov(
            self.center_cov[:self.center_dimension, :self.center_dimension]
            + self.center_measure_cov)
        distance = float(innovation @ np.linalg.solve(covariance, innovation))
        return distance, innovation, covariance

    def miss(self, evidence: _SceneEvidence) -> None:
        prior = self.existence
        detection_probability = evidence.detection_probability
        denominator = max(1.0 - prior * detection_probability, EPS)
        self.existence = float(np.clip(
            prior * (1.0 - detection_probability) / denominator, EPS, 1.0 - EPS))
        evidence.observe_miss(prior)

    def update(self, detection: dict, measurement_dims: np.ndarray,
               evidence: _SceneEvidence, config: TrainingFreeConfig) -> None:
        distance, innovation, innovation_cov = self.association(detection)
        prior_existence = self.existence
        # Clutter intensity is declared per horizontal search area, so the
        # existence Bayes factor must use the matching 2-D spatial density even
        # when the motion/association state also models z.
        target_likelihood = evidence.detection_probability * _gaussian_density(
            innovation[:2], innovation_cov[:2, :2])
        clutter_likelihood = max(evidence.clutter_density, EPS)
        numerator = prior_existence * target_likelihood
        self.existence = float(np.clip(
            numerator / max(numerator + (1.0 - prior_existence) * clutter_likelihood, EPS),
            EPS, 1.0 - EPS))
        evidence.observe_hit(prior_existence)

        dimension = self.center_dimension
        old_velocity = self.center_mean[dimension:].copy()
        measurement_matrix = np.zeros((dimension, 2 * dimension), dtype=float)
        measurement_matrix[:, :dimension] = np.eye(dimension)
        residual_cov = _as_cov(
            measurement_matrix @ self.center_cov @ measurement_matrix.T
            + self.center_measure_cov)
        gain = self.center_cov @ measurement_matrix.T @ np.linalg.inv(residual_cov)
        self.center_mean += gain @ innovation
        identity = np.eye(2 * dimension)
        ikh = identity - gain @ measurement_matrix
        self.center_cov = _as_cov(
            ikh @ self.center_cov @ ikh.T + gain @ self.center_measure_cov @ gain.T)
        self.center_innovations.update(innovation)
        self.center_measure_cov = self.center_innovations.covariance()
        self.velocity_steps.update(self.center_mean[dimension:] - old_velocity)
        self.center_process_cov = self.velocity_steps.covariance()

        dim_measurement = np.log(np.maximum(measurement_dims, 1e-6))
        dim_innovation = dim_measurement - self.dim_mean
        dim_floor = self._dimension_floor(detection, config)
        self.dim_innovations.update(dim_innovation)
        dim_measure_cov = _as_cov(self.dim_innovations.covariance() + dim_floor)
        dim_sum = _as_cov(self.dim_cov + dim_measure_cov)
        if config.robust_temporal_size:
            squared_distance = float(
                dim_innovation @ np.linalg.solve(dim_sum, dim_innovation))
            # Multivariate Huber update with its transition fixed at the 99%
            # chi-square boundary; outliers are downweighted, not discarded.
            if squared_distance > CHI2_3_99:
                weight = sqrt(CHI2_3_99 / max(squared_distance, EPS))
                dim_measure_cov = _as_cov(dim_measure_cov / max(weight, EPS))
                dim_sum = _as_cov(self.dim_cov + dim_measure_cov)
        dim_gain = self.dim_cov @ np.linalg.inv(dim_sum)
        self.dim_mean += dim_gain @ dim_innovation
        dim_ik = np.eye(3) - dim_gain
        self.dim_cov = _as_cov(
            dim_ik @ self.dim_cov @ dim_ik.T + dim_gain @ dim_measure_cov @ dim_gain.T)

        measured_rotation = project_so3(detection["R"])
        step = rotation_innovation(self.previous_measured_rotation, measured_rotation,
                                   fold_z_pi=config.fold_rotation_z_pi)
        self.rotation_steps.update(step)
        self.previous_measured_rotation = measured_rotation
        rotation_residual = rotation_innovation(
            self.rotation_mean, measured_rotation, fold_z_pi=config.fold_rotation_z_pi)
        rotation_measure_cov = self.rotation_innovations.covariance()
        rotation_sum = _as_cov(self.rotation_cov + rotation_measure_cov)
        rotation_distance = float(
            rotation_residual @ np.linalg.solve(rotation_sum, rotation_residual))
        if rotation_distance <= config.rotation_chi2:
            self.rotation_innovations.update(rotation_residual)
            rotation_measure_cov = self.rotation_innovations.covariance()
            rotation_sum = _as_cov(self.rotation_cov + rotation_measure_cov)
            rotation_gain = self.rotation_cov @ np.linalg.inv(rotation_sum)
            self.rotation_mean = self.rotation_mean @ so3_exp(rotation_gain @ rotation_residual)
            orthogonality = float(np.linalg.norm(self.rotation_mean.T @ self.rotation_mean - np.eye(3)))
            if orthogonality > 1e-6 or abs(float(np.linalg.det(self.rotation_mean)) - 1.0) > 1e-6:
                self.rotation_mean = project_so3(self.rotation_mean)
            rotation_ik = np.eye(3) - rotation_gain
            self.rotation_cov = _as_cov(
                rotation_ik @ self.rotation_cov @ rotation_ik.T
                + rotation_gain @ rotation_measure_cov @ rotation_gain.T)

        self.hits += 1
        self.missed = 0

    def apply(self, detection: dict, point_dims: np.ndarray, point_debug: dict,
              config: TrainingFreeConfig) -> dict:
        output = dict(detection)
        raw_dims = np.array([detection[key] for key in ("l", "w", "h")], dtype=float)
        dimensions = raw_dims.copy()
        if config.refine_point_size:
            dimensions = point_dims.copy()
        if config.refine_temporal_size:
            dimensions = np.exp(self.dim_mean)
        if config.bidirectional_size:
            resolution = _local_axis_resolution(detection["R"], config.voxel_size_xyz)
            dimensions = np.clip(dimensions,
                                 np.maximum(raw_dims - resolution, resolution),
                                 raw_dims + resolution)
        else:
            dimensions = np.minimum(raw_dims, dimensions)
        if config.refine_point_size or config.refine_temporal_size:
            for key, value in zip(("l", "w", "h"), dimensions):
                output[key] = float(value)
        if config.refine_rotation and self.hits >= 2:
            output["R"] = self.rotation_mean.copy()
        if config.refine_center:
            keys = ("x", "y", "z") if self.center_dimension == 3 else ("x", "y")
            for key, value in zip(keys, self.center_mean[:self.center_dimension]):
                output[key] = float(value)
        if config.refine_existence:
            if config.neutral_logodds_score:
                temporal_log_bayes = (
                    _logit(self.existence) - _logit(config.birth_existence))
                output["score"] = _sigmoid(
                    _logit(float(detection["score"])) + temporal_log_bayes)
            else:
                output["score"] = float(np.clip(
                    detection["score"] * self.existence, 0.0, 1.0))
        if config.emit_debug_fields:
            output.update({
                "tf_track_id": self.track_id,
                "tf_hits": self.hits,
                "tf_missed": self.missed,
                "tf_existence": self.existence,
                "tf_dim_std": np.sqrt(np.diag(self.dim_cov)).tolist(),
                "tf_rotation_std_deg": np.degrees(np.sqrt(np.diag(self.rotation_cov))).tolist(),
                "tf_point_support": point_debug,
                "tf_raw_score": float(detection["score"]),
                "tf_recovered_low_score": bool(
                    detection["score"] < config.canonical_score_threshold),
            })
        return output


class TrainingFreeBoxRefiner:
    """Scene-local strict training-free refinement state machine."""

    def __init__(self, config: TrainingFreeConfig | None = None):
        self.config = config or TrainingFreeConfig()
        self.reset()

    def reset(self) -> None:
        self.tracks: list[_TrainingFreeTrack] = []
        self.next_track_id = 0
        self.evidence = _SceneEvidence(self.config)

    def process_frame(self, detections: list[dict], points: np.ndarray | None = None,
                      dt: float = 1.0,
                      point_measurements: list[tuple[np.ndarray, dict]] | None = None
                      ) -> list[dict]:
        detections = [dict(detection) for detection in detections]
        if point_measurements is None:
            if self.config.refine_point_size:
                point_measurements = [
                    point_support_dimensions(detection, points, self.config)
                    for detection in detections
                ]
            else:
                point_measurements = [
                    (np.array([detection[key] for key in ("l", "w", "h")], dtype=float), {})
                    for detection in detections
                ]
        if len(point_measurements) != len(detections):
            raise ValueError("point_measurements must align one-to-one with detections")
        for track in self.tracks:
            track.predict(dt, self.evidence, self.config)

        assignments = {}
        assigned_tracks = set()

        high_indices = [index for index, detection in enumerate(detections)
                        if detection["score"] >= self.config.canonical_score_threshold]
        low_indices = [index for index, detection in enumerate(detections)
                       if detection["score"] < self.config.canonical_score_threshold]

        def assign(track_indices: list[int], detection_indices: list[int], gate: float) -> None:
            if not track_indices or not detection_indices:
                return
            cost = np.full((len(track_indices), len(detection_indices)), np.inf, dtype=float)
            for local_track, track_index in enumerate(track_indices):
                for local_detection, detection_index in enumerate(detection_indices):
                    distance, _, _ = self.tracks[track_index].association(
                        detections[detection_index])
                    if distance <= gate:
                        cost[local_track, local_detection] = distance
            for local_track, local_detection in _hungarian_dense(cost):
                if np.isfinite(cost[local_track, local_detection]):
                    track_index = track_indices[local_track]
                    detection_index = detection_indices[local_detection]
                    assignments[detection_index] = track_index
                    assigned_tracks.add(track_index)

        high_gate = (CHI2_3_99 if self.config.center_3d
                     else self.config.association_chi2)
        assign(list(range(len(self.tracks))), high_indices, high_gate)
        if self.config.recover_low_score:
            eligible_tracks = [
                index for index, track in enumerate(self.tracks)
                if index not in assigned_tracks
                and track.hits >= self.config.recovery_confirmation_hits
            ]
            assign(eligible_tracks, low_indices, self.config.recovery_chi2)

        for track_index, track in enumerate(self.tracks):
            if track_index not in assigned_tracks:
                track.miss(self.evidence)

        output = []
        new_count = 0
        for detection_index, detection in enumerate(detections):
            point_dims, point_debug = point_measurements[detection_index]
            measurement_dims = point_dims if self.config.refine_point_size else np.array(
                [detection[key] for key in ("l", "w", "h")], dtype=float)
            if detection_index in assignments:
                track = self.tracks[assignments[detection_index]]
                track.update(detection, measurement_dims, self.evidence, self.config)
            else:
                if detection["score"] < self.config.canonical_score_threshold:
                    # Low peaks are evidence for an already confirmed target;
                    # they can neither create a track nor be emitted alone.
                    continue
                track = _TrainingFreeTrack(
                    self.next_track_id, detection, measurement_dims, self.config)
                self.next_track_id += 1
                self.tracks.append(track)
                new_count += 1
            output.append(track.apply(detection, point_dims, point_debug, self.config))

        self.evidence.observe_frame(new_count)
        self.tracks = [
            track for track in self.tracks
            if track.missed <= self.config.max_missed_frames
        ]
        return output

    def process_scene(self, frames: list[list[dict]], points: list[np.ndarray] | None = None,
                      frame_ids: list[int] | None = None) -> list[list[dict]]:
        self.reset()
        output = []
        previous_frame = None
        for index, detections in enumerate(frames):
            frame = frame_ids[index] if frame_ids is not None else index
            dt = 1.0 if previous_frame is None else max(float(frame - previous_frame), 1.0)
            frame_points = None if points is None else points[index]
            output.append(self.process_frame(detections, points=frame_points, dt=dt))
            previous_frame = frame
        return output


ABLATION_CONFIGS = {
    "raw": None,
    "association_only": TrainingFreeConfig(),
    "existence_only": TrainingFreeConfig(refine_existence=True),
    "point_size_only": TrainingFreeConfig(refine_point_size=True),
    "temporal_size_only": TrainingFreeConfig(refine_temporal_size=True),
    "training_free_size": TrainingFreeConfig(
        refine_point_size=True, refine_temporal_size=True),
    "rotation_only": TrainingFreeConfig(refine_rotation=True),
    "existence_size": TrainingFreeConfig(
        refine_existence=True, refine_point_size=True, refine_temporal_size=True),
    "size_rotation": TrainingFreeConfig(
        refine_point_size=True, refine_temporal_size=True, refine_rotation=True),
    "full": TrainingFreeConfig(
        refine_existence=True, refine_point_size=True,
        refine_temporal_size=True, refine_rotation=True),
}


V3C_ABLATION_CONFIGS = {
    "raw": None,
    "legacy_training_free_size": TrainingFreeConfig(
        refine_point_size=True, refine_temporal_size=True),
    "neutral_score": TrainingFreeConfig(
        refine_existence=True, neutral_logodds_score=True),
    "robust_size": TrainingFreeConfig(
        refine_point_size=True, refine_temporal_size=True,
        robust_temporal_size=True, bidirectional_size=True),
    "center3d": TrainingFreeConfig(
        refine_center=True, center_3d=True),
    "recovery_score": TrainingFreeConfig(
        refine_existence=True, neutral_logodds_score=True,
        recover_low_score=True, center_3d=True, emit_debug_fields=True),
    "recovery_size_center": TrainingFreeConfig(
        refine_existence=True, neutral_logodds_score=True,
        recover_low_score=True, refine_point_size=True,
        refine_temporal_size=True, robust_temporal_size=True,
        bidirectional_size=True, refine_center=True, center_3d=True,
        emit_debug_fields=True),
}


# Frozen 2026-09-11 after the two-seed neutral-score x size factorial.  Center
# experiments import this exact object instead of reconstructing a nearly equal
# configuration in their runner.  ``emit_debug_fields`` only exposes track ids
# needed by downstream center smoothing; it does not change any box field.
FROZEN_REFINEMENT_BASELINE = TrainingFreeConfig(
    refine_existence=True,
    neutral_logodds_score=True,
    refine_point_size=True,
    refine_temporal_size=True,
    robust_temporal_size=True,
    bidirectional_size=True,
    emit_debug_fields=True,
)
