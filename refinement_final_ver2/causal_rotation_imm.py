"""Strictly causal SO(3) interacting multiple-model rotation refinement.

The implementation follows the standard IMM cycle -- interaction, mode-
conditioned filtering, likelihood update, and combination -- while performing
orientation means and differences in the tangent space of SO(3).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from bayesian_box_filter import (_as_cov, project_so3, rotation_innovation,
                                 so3_exp)
from causal_bayesian_center import FrozenCausalOnlineSuite
from training_free_filter import CHI2_3_99, EPS, RunningCovariance


@dataclass(frozen=True)
class RotationIMMConfig:
    huber_chi2: float = CHI2_3_99
    fold_rotation_z_pi: bool = True
    tangent_floor_deg: float = 0.1
    # A conventional sticky Markov prior. It is declared before evaluation and
    # is not estimated from train/validation trajectories.
    mode_persistence: float = 0.90
    # Stationary, constant-angular-velocity, and maneuver process regimes.
    process_scales: tuple[float, float, float] = (0.25, 1.0, 16.0)


ROTATION_IMM_CONFIG = RotationIMMConfig()


@dataclass
class _ModeState:
    rotation: np.ndarray
    angular_velocity: np.ndarray
    covariance: np.ndarray


def _so3_weighted_mean(rotations: list[np.ndarray], weights: np.ndarray,
                       fold_z_pi: bool) -> np.ndarray:
    """Local Karcher mean used by IMM interaction and output combination."""
    mean = rotations[int(np.argmax(weights))].copy()
    for _ in range(8):
        step = sum((float(weight) * rotation_innovation(
            mean, rotation, fold_z_pi=fold_z_pi)
                    for weight, rotation in zip(weights, rotations)),
                   np.zeros(3, dtype=float))
        if float(np.linalg.norm(step)) < 1e-10:
            break
        mean = project_so3(mean @ so3_exp(step))
    return mean


def _normal_log_likelihood(residual: np.ndarray, covariance: np.ndarray) -> float:
    covariance = _as_cov(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return -60.0
    nis = float(residual @ np.linalg.solve(covariance, residual))
    return float(max(-0.5 * (3.0 * np.log(2.0 * np.pi) + logdet + nis), -60.0))


class CausalRotationIMMTrack:
    """Stationary/CAV/maneuver IMM with right-multiplicative SO(3) errors."""

    def __init__(self, rotation: np.ndarray,
                 config: RotationIMMConfig = ROTATION_IMM_CONFIG):
        self.config = config
        rotation = project_so3(rotation)
        covariance = np.eye(6, dtype=float) * (pi / 4.0) ** 2
        self.modes = [_ModeState(rotation.copy(), np.zeros(3), covariance.copy())
                      for _ in config.process_scales]
        self.mode_probabilities = np.full(len(self.modes), 1.0 / len(self.modes))
        off_diagonal = ((1.0 - config.mode_persistence)
                        / max(len(self.modes) - 1, 1))
        self.transition_probabilities = np.full(
            (len(self.modes), len(self.modes)), off_diagonal)
        np.fill_diagonal(self.transition_probabilities, config.mode_persistence)
        floor = np.eye(3) * np.radians(config.tangent_floor_deg) ** 2
        self.measurement_residuals = RunningCovariance(3, floor)
        self.angular_accelerations = RunningCovariance(3, floor)
        self.previous_measurement = rotation.copy()
        self.previous_observed_rate: np.ndarray | None = None
        self.last_frame: int | None = None
        self.hits = 0

    def _interaction(self) -> tuple[list[_ModeState], np.ndarray]:
        predicted_probabilities = self.mode_probabilities @ self.transition_probabilities
        mixed = []
        for destination in range(len(self.modes)):
            weights = (self.mode_probabilities
                       * self.transition_probabilities[:, destination])
            weights /= max(float(predicted_probabilities[destination]), EPS)
            rotation = _so3_weighted_mean(
                [mode.rotation for mode in self.modes], weights,
                self.config.fold_rotation_z_pi)
            angular_velocity = sum(
                (float(weight) * mode.angular_velocity
                 for weight, mode in zip(weights, self.modes)),
                np.zeros(3, dtype=float))
            covariance = np.zeros((6, 6), dtype=float)
            for weight, mode in zip(weights, self.modes):
                orientation_delta = rotation_innovation(
                    rotation, mode.rotation,
                    fold_z_pi=self.config.fold_rotation_z_pi)
                delta = np.r_[orientation_delta,
                              mode.angular_velocity - angular_velocity]
                covariance += float(weight) * (mode.covariance
                                                + np.outer(delta, delta))
            mixed.append(_ModeState(rotation, angular_velocity,
                                    _as_cov(covariance)))
        return mixed, predicted_probabilities

    @staticmethod
    def _dynamics(mode_index: int, dt: float) -> np.ndarray:
        transition = np.eye(6, dtype=float)
        if mode_index == 0:  # stationary: discard inherited angular rate
            transition[:3, 3:] = 0.0
            transition[3:, 3:] = 0.0
        else:
            transition[:3, 3:] = dt * np.eye(3)
        return transition

    def _predict(self, state: _ModeState, mode_index: int, dt: float,
                 acceleration_covariance: np.ndarray) -> _ModeState:
        transition = self._dynamics(mode_index, dt)
        velocity = (np.zeros(3, dtype=float) if mode_index == 0
                    else state.angular_velocity.copy())
        rotation = project_so3(state.rotation @ so3_exp(velocity * dt))
        scale = self.config.process_scales[mode_index]
        q = scale * np.block([
            [0.25 * dt**4 * acceleration_covariance,
             0.5 * dt**3 * acceleration_covariance],
            [0.5 * dt**3 * acceleration_covariance,
             dt**2 * acceleration_covariance],
        ])
        covariance = _as_cov(transition @ state.covariance @ transition.T + q)
        return _ModeState(rotation, velocity, covariance)

    def _update_mode(self, state: _ModeState, measured_rotation: np.ndarray,
                     measurement_covariance: np.ndarray
                     ) -> tuple[_ModeState, float, float, float]:
        residual = rotation_innovation(
            state.rotation, measured_rotation,
            fold_z_pi=self.config.fold_rotation_z_pi)
        h = np.zeros((3, 6), dtype=float)
        h[:, :3] = np.eye(3)
        innovation_covariance = _as_cov(h @ state.covariance @ h.T
                                        + measurement_covariance)
        nis = float(residual @ np.linalg.solve(innovation_covariance, residual))
        robust_weight = 1.0
        effective_measurement_covariance = measurement_covariance
        if nis > self.config.huber_chi2:
            robust_weight = sqrt(self.config.huber_chi2 / max(nis, EPS))
            effective_measurement_covariance = _as_cov(
                measurement_covariance / max(robust_weight, EPS))
            innovation_covariance = _as_cov(
                h @ state.covariance @ h.T + effective_measurement_covariance)
        gain = state.covariance @ h.T @ np.linalg.inv(innovation_covariance)
        correction = gain @ residual
        rotation = project_so3(state.rotation @ so3_exp(correction[:3]))
        velocity = state.angular_velocity + correction[3:]
        ikh = np.eye(6) - gain @ h
        covariance = _as_cov(
            ikh @ state.covariance @ ikh.T
            + gain @ effective_measurement_covariance @ gain.T)
        likelihood = _normal_log_likelihood(residual, innovation_covariance)
        return (_ModeState(rotation, velocity, covariance), likelihood, nis,
                robust_weight)

    def _combined_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotation = _so3_weighted_mean(
            [mode.rotation for mode in self.modes], self.mode_probabilities,
            self.config.fold_rotation_z_pi)
        velocity = sum((float(weight) * mode.angular_velocity
                        for weight, mode in zip(self.mode_probabilities, self.modes)),
                       np.zeros(3, dtype=float))
        covariance = np.zeros((6, 6), dtype=float)
        for weight, mode in zip(self.mode_probabilities, self.modes):
            delta = np.r_[rotation_innovation(
                rotation, mode.rotation,
                fold_z_pi=self.config.fold_rotation_z_pi),
                mode.angular_velocity - velocity]
            covariance += float(weight) * (mode.covariance + np.outer(delta, delta))
        return rotation, velocity, _as_cov(covariance)

    def update(self, frame_id: int, measured_rotation: np.ndarray,
               measurement_covariance: np.ndarray | None = None
               ) -> tuple[np.ndarray, dict]:
        measured_rotation = project_so3(measured_rotation)
        if self.last_frame is None:
            self.last_frame = int(frame_id)
            self.hits = 1
            return measured_rotation.copy(), self.diagnostics(None, None)
        dt = max(float(frame_id - self.last_frame), 1.0)
        observed_step = rotation_innovation(
            self.previous_measurement, measured_rotation,
            fold_z_pi=self.config.fold_rotation_z_pi)
        observed_rate = observed_step / dt
        if self.hits == 1:
            for mode in self.modes:
                mode.rotation = measured_rotation.copy()
                mode.angular_velocity = observed_rate.copy()
            self.previous_observed_rate = observed_rate
            self.previous_measurement = measured_rotation.copy()
            self.last_frame = int(frame_id)
            self.hits = 2
            return measured_rotation.copy(), self.diagnostics(None, None)

        acceleration = (observed_rate - self.previous_observed_rate) / dt
        self.angular_accelerations.update(acceleration)
        mixed, predicted_probabilities = self._interaction()
        predicted = [self._predict(
            state, index, dt, self.angular_accelerations.covariance())
                     for index, state in enumerate(mixed)]
        predicted_rotation = _so3_weighted_mean(
            [state.rotation for state in predicted], predicted_probabilities,
            self.config.fold_rotation_z_pi)
        if measurement_covariance is None:
            self.measurement_residuals.update(rotation_innovation(
                predicted_rotation, measured_rotation,
                fold_z_pi=self.config.fold_rotation_z_pi))
            measurement_covariance = self.measurement_residuals.covariance()
        else:
            measurement_covariance = _as_cov(measurement_covariance)

        updated, log_likelihoods, nis_values, robust_weights = [], [], [], []
        for state in predicted:
            posterior, log_likelihood, nis, weight = self._update_mode(
                state, measured_rotation, measurement_covariance)
            updated.append(posterior)
            log_likelihoods.append(log_likelihood)
            nis_values.append(nis)
            robust_weights.append(weight)
        log_posterior = np.log(np.maximum(predicted_probabilities, EPS)) + log_likelihoods
        log_posterior -= float(np.max(log_posterior))
        probabilities = np.exp(log_posterior)
        self.mode_probabilities = probabilities / max(float(np.sum(probabilities)), EPS)
        self.modes = updated

        rotation, _, _ = self._combined_state()
        self.previous_observed_rate = observed_rate
        self.previous_measurement = measured_rotation.copy()
        self.last_frame = int(frame_id)
        self.hits += 1
        weighted_nis = float(self.mode_probabilities @ np.asarray(nis_values))
        weighted_robust = float(self.mode_probabilities @ np.asarray(robust_weights))
        return rotation, self.diagnostics(weighted_nis, weighted_robust)

    def diagnostics(self, nis: float | None, robust_weight: float | None) -> dict:
        _, velocity, covariance = self._combined_state()
        return {
            "hits": self.hits,
            "mode_probabilities": self.mode_probabilities.tolist(),
            "angular_speed_deg": float(np.degrees(np.linalg.norm(velocity))),
            "orientation_std_deg": np.degrees(
                np.sqrt(np.diag(covariance[:3, :3]))).tolist(),
            "nis": nis,
            "robust_weight": robust_weight,
        }


class CausalRotationIMMSuite:
    """Frozen score/size/center suite plus independent SO(3) IMM rotation."""

    method_version = "v3-l-causal-so3-imm-1"

    def __init__(self, config: RotationIMMConfig = ROTATION_IMM_CONFIG):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.frozen = FrozenCausalOnlineSuite()
        self.rotation_tracks: dict[int, CausalRotationIMMTrack] = {}

    def process_frame(self, detections: list[dict], points: np.ndarray | None = None,
                      frame_id: int | None = None,
                      point_measurements: list[tuple[np.ndarray, dict]] | None = None
                      ) -> list[dict]:
        frozen = self.frozen.process_frame(
            detections, points, frame_id, point_measurements)
        output = []
        for source in frozen:
            box = dict(source)
            track_id = int(box["tf_track_id"])
            if track_id not in self.rotation_tracks:
                self.rotation_tracks[track_id] = CausalRotationIMMTrack(
                    box["R"], self.config)
            rotation, diagnostic = self.rotation_tracks[track_id].update(
                int(self.frozen.frame_id), box["R"])
            box["R"] = rotation
            box["rotation_imm"] = diagnostic
            output.append(box)
        active_ids = {track.track_id for track in self.frozen.box_refiner.tracks}
        self.rotation_tracks = {
            key: value for key, value in self.rotation_tracks.items()
            if key in active_ids
        }
        return output
