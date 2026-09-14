"""Causal Bayesian multiple-model center refinement.

This is a Rao-Blackwellised interacting multiple-model (IMM) filter.  Discrete
motion and heatmap-covariance hypotheses are marginalised, while each
conditional continuous state is updated by a Kalman filter.  Every output uses
only observations at or before that frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from math import log, pi

import numpy as np

from bayesian_box_filter import _as_cov
from training_free_filter import (CHI2_2_95, EPS, FROZEN_REFINEMENT_BASELINE,
                                  TrainingFreeBoxRefiner)


MOTION_MODES = ("stationary", "constant_velocity", "maneuver")


@dataclass(frozen=True)
class CausalBayesianCenterConfig:
    cell_xy: tuple[float, float] = (0.2, 0.2)
    voxel_z: float = 0.5
    motion_modes: tuple[str, ...] = MOTION_MODES
    # 0=fixed sensor floor, 1=full heatmap shape.  The middle hypothesis is the
    # inverse 95% chi-square radius, not a validation-selected scale.
    heatmap_shape_scales: tuple[float, ...] = (0.0, 1.0 / CHI2_2_95, 1.0)
    # Reuse the already declared per-frame track survival probability as the
    # mode-persistence prior.  Remaining mass is symmetric across other modes.
    mode_stay_probability: float = 0.99
    # Initial velocity variance in units of the quantization-floor variance.
    # Default 4 preserves the frozen candidate; 12 means one full sensor cell
    # of unknown displacement per frame. No age threshold or residual fit.
    birth_velocity_variance_scale: float = 4.0
    # Multipliers for process covariance in world x, world y and z. Defaults
    # preserve the frozen suite exactly; Stage-T may estimate them from causal
    # innovations without changing the sensor measurement covariance.
    process_noise_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)


# Two-seed primary candidate.  The frozen suite baseline remains unchanged;
# this constant is the center component to compose on top of it in future
# causal-online evaluations.
PRIMARY_CAUSAL_CENTER_CONFIG = CausalBayesianCenterConfig(
    heatmap_shape_scales=(0.0,),
)

BIRTH_UNCERTAINTY_CENTER_CONFIG = CausalBayesianCenterConfig(
    heatmap_shape_scales=(0.0,),
    birth_velocity_variance_scale=12.0,
)


class FrozenCausalOnlineSuite:
    """Frozen v3-e baseline composed as a true frame-by-frame online module."""

    method_version = "v3-e-frozen-causal-online-suite-1"

    def __init__(self, emit_center_diagnostics: bool = False,
                 center_config: CausalBayesianCenterConfig =
                 PRIMARY_CAUSAL_CENTER_CONFIG):
        self.emit_center_diagnostics = bool(emit_center_diagnostics)
        self.center_config = center_config
        self.reset()

    def reset(self) -> None:
        self.box_refiner = TrainingFreeBoxRefiner(FROZEN_REFINEMENT_BASELINE)
        self.center_tracks: dict[int, CausalBayesianCenterTrack] = {}
        self.frame_id: int | None = None

    def process_frame(self, detections: list[dict], points: np.ndarray | None = None,
                      frame_id: int | None = None,
                      point_measurements: list[tuple[np.ndarray, dict]] | None = None
                      ) -> list[dict]:
        if frame_id is None:
            frame_id = 0 if self.frame_id is None else self.frame_id + 1
        dt = 1.0 if self.frame_id is None else max(float(frame_id - self.frame_id), 1.0)
        self.frame_id = int(frame_id)
        baseline = self.box_refiner.process_frame(
            detections, points=points, dt=dt, point_measurements=point_measurements)
        output = []
        for source in baseline:
            box = dict(source)
            track_id = int(box["tf_track_id"])
            if track_id not in self.center_tracks:
                self.center_tracks[track_id] = CausalBayesianCenterTrack(
                    box, self.center_config)
            state, diagnostic = self.center_tracks[track_id].update(frame_id, box)
            box["x"], box["y"], box["z"] = (
                float(state[0]), float(state[1]), float(state[4]))
            box["center_bayes_motion_weights"] = diagnostic["motion_weights"]
            box["center_bayes_scale_weights"] = diagnostic["scale_weights"]
            if self.emit_center_diagnostics:
                box["center_bayes_diagnostic"] = diagnostic
            output.append(box)
        # Track ids are monotonic and never reused.  Dropping dead center states
        # bounds online memory without affecting any future output.
        active_ids = {track.track_id for track in self.box_refiner.tracks}
        self.center_tracks = {
            track_id: track for track_id, track in self.center_tracks.items()
            if track_id in active_ids
        }
        return output


def _log_gaussian(residual: np.ndarray, covariance: np.ndarray) -> float:
    covariance = _as_cov(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return -np.inf
    distance = float(residual @ np.linalg.solve(covariance, residual))
    return -0.5 * (len(residual) * log(2.0 * pi) + logdet + distance)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return -np.inf
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _mode_transition(count: int, stay: float) -> np.ndarray:
    if count == 1:
        return np.ones((1, 1), dtype=float)
    switch = (1.0 - stay) / (count - 1)
    matrix = np.full((count, count), switch, dtype=float)
    np.fill_diagonal(matrix, stay)
    return matrix


def _measurement_parts(box: dict, config: CausalBayesianCenterConfig):
    floor = np.diag([
        config.cell_xy[0] ** 2 / 12.0,
        config.cell_xy[1] ** 2 / 12.0,
        config.voxel_z ** 2 / 12.0,
    ])
    supplied = np.asarray(box.get("center_measurement_cov", floor), dtype=float)
    # A separately constructed Gaussian factor (for example, the product of
    # detector and point-registration likelihoods) may legitimately be more
    # informative than either sensor-floor factor alone. Frozen detector-only
    # behavior remains unchanged because this flag is absent there.
    if box.get("center_measurement_cov_absolute", False):
        return _as_cov(supplied), np.zeros((3, 3), dtype=float)
    shape = supplied - floor
    shape[2, :] = 0.0
    shape[:, 2] = 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (shape + shape.T))
    shape = eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return floor, shape


def _transition_and_noise(mode: str, dt: float,
                          config: CausalBayesianCenterConfig):
    """Five-state [x,y,vx,vy,z] dynamics; Z is always a separate random walk."""
    transition = np.eye(5)
    process_scale = np.asarray(config.process_noise_scale_xyz, dtype=float)
    floor_xy = np.diag(np.square(config.cell_xy) * process_scale[:2]) / 12.0
    q = np.zeros((5, 5), dtype=float)
    if mode == "stationary":
        transition[2, 2] = 0.0
        transition[3, 3] = 0.0
        q[:2, :2] = floor_xy * max(dt, 1.0)
        q[2:4, 2:4] = floor_xy
    elif mode in ("constant_velocity", "maneuver"):
        transition[0, 2] = dt
        transition[1, 3] = dt
        acceleration = np.diag(np.square(config.cell_xy) * process_scale[:2])
        if mode == "maneuver":
            acceleration *= CHI2_2_95
        q[:2, :2] = 0.25 * dt**4 * acceleration
        q[:2, 2:4] = 0.5 * dt**3 * acceleration
        q[2:4, :2] = 0.5 * dt**3 * acceleration
        q[2:4, 2:4] = dt**2 * acceleration
    else:
        raise ValueError(f"unknown motion mode: {mode}")
    # Z has no heatmap plane or justified velocity observation.  Keeping it a
    # random walk avoids importing XY constant-velocity assumptions into depth.
    q[4, 4] = (config.voxel_z**2 / 12.0 * process_scale[2]
               * max(dt, 1.0))
    return transition, _as_cov(q)


class CausalBayesianCenterTrack:
    def __init__(self, box: dict, config: CausalBayesianCenterConfig):
        self.config = config
        self.modes = tuple(config.motion_modes)
        self.scales = tuple(float(value) for value in config.heatmap_shape_scales)
        if not self.modes or not self.scales:
            raise ValueError("at least one motion mode and covariance hypothesis required")
        unknown = set(self.modes) - set(MOTION_MODES)
        if unknown:
            raise ValueError(f"unknown motion modes: {sorted(unknown)}")
        if any(value < 0.0 for value in self.scales):
            raise ValueError("heatmap shape scales must be non-negative")
        if (len(config.process_noise_scale_xyz) != 3
                or any(value < 0.0 for value in config.process_noise_scale_xyz)):
            raise ValueError("process noise scales must be three non-negative values")
        position = np.array([box["x"], box["y"], box["z"]], dtype=float)
        initial = np.array([position[0], position[1], 0.0, 0.0, position[2]])
        floor, _ = _measurement_parts(box, config)
        covariance = np.zeros((5, 5), dtype=float)
        covariance[0, 0], covariance[1, 1], covariance[4, 4] = np.diag(floor)
        covariance[2, 2] = floor[0, 0] * config.birth_velocity_variance_scale
        covariance[3, 3] = floor[1, 1] * config.birth_velocity_variance_scale
        shape = (len(self.modes), len(self.scales))
        self.means = np.broadcast_to(initial, shape + (5,)).copy()
        self.covariances = np.broadcast_to(covariance, shape + (5, 5)).copy()
        self.weights = np.full(shape, 1.0 / (shape[0] * shape[1]), dtype=float)
        self.last_frame: int | None = None

    def update(self, frame_id: int, box: dict,
               velocity_measurement_xy: np.ndarray | None = None,
               velocity_covariance_xy: np.ndarray | None = None
               ) -> tuple[np.ndarray, dict]:
        if self.last_frame is None:
            self.last_frame = int(frame_id)
            return self.posterior_mean(), self.diagnostics()
        dt = max(float(frame_id - self.last_frame), 1.0)
        self.last_frame = int(frame_id)
        observation = np.array([box["x"], box["y"], box["z"]], dtype=float)
        h = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1]], dtype=float)
        floor, heatmap_shape = _measurement_parts(box, self.config)
        transition_probability = _mode_transition(
            len(self.modes), self.config.mode_stay_probability)

        new_means = np.zeros_like(self.means)
        new_covariances = np.zeros_like(self.covariances)
        predicted_means = np.zeros_like(self.means)
        predicted_covariances = np.zeros_like(self.covariances)
        predicted_weights = np.zeros_like(self.weights)
        log_weights = np.full_like(self.weights, -np.inf)
        log_xy_evidence_terms = []
        minimum_xy_nis = np.inf
        minimum_velocity_nis = np.inf
        if (velocity_measurement_xy is None) != (velocity_covariance_xy is None):
            raise ValueError("velocity measurement and covariance must be supplied together")
        if velocity_measurement_xy is not None:
            velocity_measurement_xy = np.asarray(velocity_measurement_xy, dtype=float)
            velocity_covariance_xy = _as_cov(
                np.asarray(velocity_covariance_xy, dtype=float))
            if velocity_measurement_xy.shape != (2,) or velocity_covariance_xy.shape != (2, 2):
                raise ValueError("velocity factor must be 2-D")
        velocity_h = np.array([[0, 0, 1, 0, 0],
                               [0, 0, 0, 1, 0]], dtype=float)
        for scale_index, scale in enumerate(self.scales):
            source_weights = self.weights[:, scale_index]
            measurement_cov = _as_cov(floor + scale * heatmap_shape)
            for destination, mode in enumerate(self.modes):
                contributions = source_weights * transition_probability[:, destination]
                prior_weight = float(np.sum(contributions))
                mixing = contributions / max(prior_weight, EPS)
                mixed_mean = np.sum(
                    mixing[:, None] * self.means[:, scale_index], axis=0)
                mixed_cov = np.zeros((5, 5), dtype=float)
                for source in range(len(self.modes)):
                    delta = self.means[source, scale_index] - mixed_mean
                    mixed_cov += mixing[source] * (
                        self.covariances[source, scale_index] + np.outer(delta, delta))
                transition, process_cov = _transition_and_noise(mode, dt, self.config)
                predicted_mean = transition @ mixed_mean
                predicted_means[destination, scale_index] = predicted_mean
                predicted_weights[destination, scale_index] = prior_weight
                predicted_cov = _as_cov(
                    transition @ mixed_cov @ transition.T + process_cov)
                predicted_covariances[destination, scale_index] = predicted_cov
                residual = observation - h @ predicted_mean
                innovation_cov = _as_cov(
                    h @ predicted_cov @ h.T + measurement_cov)
                log_xy_evidence_terms.append(
                    log(max(prior_weight, EPS))
                    + _log_gaussian(residual[:2], innovation_cov[:2, :2]))
                minimum_xy_nis = min(
                    minimum_xy_nis,
                    float(residual[:2] @ np.linalg.solve(
                        innovation_cov[:2, :2], residual[:2])))
                gain = predicted_cov @ h.T @ np.linalg.inv(innovation_cov)
                posterior_mean = predicted_mean + gain @ residual
                ikh = np.eye(5) - gain @ h
                posterior_cov = _as_cov(
                    ikh @ predicted_cov @ ikh.T
                    + gain @ measurement_cov @ gain.T)
                velocity_log_evidence = 0.0
                if velocity_measurement_xy is not None:
                    velocity_residual = (
                        velocity_measurement_xy - velocity_h @ posterior_mean)
                    velocity_innovation_cov = _as_cov(
                        velocity_h @ posterior_cov @ velocity_h.T
                        + velocity_covariance_xy)
                    velocity_gain = (posterior_cov @ velocity_h.T
                                     @ np.linalg.inv(velocity_innovation_cov))
                    posterior_mean = posterior_mean + velocity_gain @ velocity_residual
                    ivh = np.eye(5) - velocity_gain @ velocity_h
                    posterior_cov = _as_cov(
                        ivh @ posterior_cov @ ivh.T
                        + velocity_gain @ velocity_covariance_xy @ velocity_gain.T)
                    velocity_log_evidence = _log_gaussian(
                        velocity_residual, velocity_innovation_cov)
                    minimum_velocity_nis = min(
                        minimum_velocity_nis,
                        float(velocity_residual @ np.linalg.solve(
                            velocity_innovation_cov, velocity_residual)))
                new_means[destination, scale_index] = posterior_mean
                new_covariances[destination, scale_index] = posterior_cov
                log_weights[destination, scale_index] = (
                    log(max(prior_weight, EPS))
                    + _log_gaussian(residual, innovation_cov)
                    + velocity_log_evidence)
        normalizer = _logsumexp(log_weights.ravel())
        if not np.isfinite(normalizer):
            self.weights.fill(1.0 / self.weights.size)
        else:
            self.weights = np.exp(log_weights - normalizer)
        self.means = new_means
        self.covariances = new_covariances
        diagnostic = self.diagnostics()
        diagnostic["log_predictive_evidence"] = float(normalizer)
        # Association clutter is expressed per horizontal area, so PDA must
        # compare it with the matching XY density rather than a 3-D density.
        diagnostic["log_predictive_evidence_xy"] = float(
            _logsumexp(np.asarray(log_xy_evidence_terms, dtype=float)))
        diagnostic["minimum_xy_nis"] = float(minimum_xy_nis)
        diagnostic["minimum_velocity_nis"] = (
            None if not np.isfinite(minimum_velocity_nis)
            else float(minimum_velocity_nis))
        predictive_mean = np.sum(
            predicted_weights[..., None] * predicted_means, axis=(0, 1))
        predictive_covariance = np.zeros((5, 5), dtype=float)
        for mode_index in range(len(self.modes)):
            for scale_index in range(len(self.scales)):
                delta = predicted_means[mode_index, scale_index] - predictive_mean
                predictive_covariance += predicted_weights[mode_index, scale_index] * (
                    predicted_covariances[mode_index, scale_index]
                    + np.outer(delta, delta))
        position_indices = np.array([0, 1, 4])
        diagnostic["predictive_center_xyz"] = [
            float(predictive_mean[0]), float(predictive_mean[1]),
            float(predictive_mean[4])]
        diagnostic["predictive_covariance_xyz"] = predictive_covariance[
            np.ix_(position_indices, position_indices)].tolist()
        diagnostic["innovation_xyz"] = (
            observation - np.asarray(diagnostic["predictive_center_xyz"])).tolist()
        return self.posterior_mean(), diagnostic

    def hypothetical_update(self, frame_id: int, box: dict):
        """Return a conditional posterior without mutating this track."""
        if self.last_frame is None:
            raise RuntimeError("a birth track has no predictive association density")
        candidate = copy.deepcopy(self)
        state, diagnostic = candidate.update(frame_id, box)
        return candidate, state, diagnostic

    def hypothetical_miss(self, frame_id: int):
        """Return the IMM predictive posterior for a missed observation."""
        if self.last_frame is None:
            raise RuntimeError("a birth track has no predictive state")
        candidate = copy.deepcopy(self)
        dt = max(float(frame_id - self.last_frame), 1.0)
        transition_probability = _mode_transition(
            len(self.modes), self.config.mode_stay_probability)
        new_means = np.zeros_like(self.means)
        new_covariances = np.zeros_like(self.covariances)
        new_weights = np.zeros_like(self.weights)
        for scale_index in range(len(self.scales)):
            source_weights = self.weights[:, scale_index]
            for destination, mode in enumerate(self.modes):
                contributions = source_weights * transition_probability[:, destination]
                prior_weight = float(np.sum(contributions))
                mixing = contributions / max(prior_weight, EPS)
                mixed_mean = np.sum(
                    mixing[:, None] * self.means[:, scale_index], axis=0)
                mixed_cov = np.zeros((5, 5), dtype=float)
                for source in range(len(self.modes)):
                    delta = self.means[source, scale_index] - mixed_mean
                    mixed_cov += mixing[source] * (
                        self.covariances[source, scale_index] + np.outer(delta, delta))
                transition, process_cov = _transition_and_noise(mode, dt, self.config)
                new_means[destination, scale_index] = transition @ mixed_mean
                new_covariances[destination, scale_index] = _as_cov(
                    transition @ mixed_cov @ transition.T + process_cov)
                new_weights[destination, scale_index] = prior_weight
        candidate.means = new_means
        candidate.covariances = new_covariances
        candidate.weights = new_weights / max(float(np.sum(new_weights)), EPS)
        candidate.last_frame = int(frame_id)
        return candidate

    @classmethod
    def moment_match(cls, weighted_tracks):
        """Collapse association hypotheses while retaining motion/scale modes."""
        weighted_tracks = [(float(weight), track) for weight, track in weighted_tracks
                           if float(weight) > 0.0]
        if not weighted_tracks:
            raise ValueError("at least one positive-weight track hypothesis required")
        total = sum(weight for weight, _ in weighted_tracks)
        weighted_tracks = [(weight / total, track) for weight, track in weighted_tracks]
        result = copy.deepcopy(weighted_tracks[0][1])
        component_weights = sum(
            branch * track.weights for branch, track in weighted_tracks)
        component_weights /= max(float(np.sum(component_weights)), EPS)
        means = np.zeros_like(result.means)
        covariances = np.zeros_like(result.covariances)
        for mode in range(len(result.modes)):
            for scale in range(len(result.scales)):
                denominator = max(float(component_weights[mode, scale]), EPS)
                conditional = [branch * track.weights[mode, scale] / denominator
                               for branch, track in weighted_tracks]
                mean = sum(weight * track.means[mode, scale]
                           for weight, (_, track) in zip(conditional, weighted_tracks))
                covariance = np.zeros((5, 5), dtype=float)
                for weight, (_, track) in zip(conditional, weighted_tracks):
                    delta = track.means[mode, scale] - mean
                    covariance += weight * (
                        track.covariances[mode, scale] + np.outer(delta, delta))
                means[mode, scale] = mean
                covariances[mode, scale] = _as_cov(covariance)
        result.weights = component_weights
        result.means = means
        result.covariances = covariances
        return result

    def posterior_mean(self) -> np.ndarray:
        return np.sum(self.weights[..., None] * self.means, axis=(0, 1))

    def diagnostics(self) -> dict:
        motion = np.sum(self.weights, axis=1)
        scales = np.sum(self.weights, axis=0)
        entropy = -float(np.sum(self.weights * np.log(np.maximum(self.weights, EPS))))
        return {
            "motion_weights": {name: float(value)
                               for name, value in zip(self.modes, motion)},
            "scale_weights": {str(value): float(weight)
                              for value, weight in zip(self.scales, scales)},
            "posterior_entropy": entropy,
        }


def refine_scene_centers_causal(frames: list[list[dict]], frame_ids: list[int],
                                config: CausalBayesianCenterConfig):
    """Apply the filter causally to frozen-baseline track assignments."""
    if len(frames) != len(frame_ids):
        raise ValueError("frame_ids must align with frames")
    tracks: dict[int, CausalBayesianCenterTrack] = {}
    output = []
    motion_totals = {name: 0.0 for name in config.motion_modes}
    scale_totals = {str(value): 0.0 for value in config.heatmap_shape_scales}
    entropy_total = 0.0
    observations = 0
    for frame_id, boxes in zip(frame_ids, frames):
        frame_output = []
        for source in boxes:
            box = dict(source)
            track_id = int(box["tf_track_id"])
            if track_id not in tracks:
                tracks[track_id] = CausalBayesianCenterTrack(box, config)
            state, diagnostic = tracks[track_id].update(frame_id, box)
            box["x"], box["y"], box["z"] = (
                float(state[0]), float(state[1]), float(state[4]))
            box["center_bayes_motion_weights"] = diagnostic["motion_weights"]
            box["center_bayes_scale_weights"] = diagnostic["scale_weights"]
            frame_output.append(box)
            for name, value in diagnostic["motion_weights"].items():
                motion_totals[name] += value
            for name, value in diagnostic["scale_weights"].items():
                scale_totals[name] += value
            entropy_total += diagnostic["posterior_entropy"]
            observations += 1
        output.append(frame_output)
    divisor = max(observations, 1)
    return output, {
        "tracks": len(tracks),
        "observations": observations,
        "mean_motion_posterior": {key: value / divisor
                                  for key, value in motion_totals.items()},
        "mean_scale_posterior": {key: value / divisor
                                 for key, value in scale_totals.items()},
        "mean_joint_entropy": entropy_total / divisor,
    }
