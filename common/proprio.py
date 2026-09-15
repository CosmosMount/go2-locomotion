"""Shared Torch proprioceptive velocity estimator for Isaac vector tasks.

The observation ABI and task projections live in ``common.observation``. This
module supplies the common Isaac-side implementation used to populate the
``estimated_body_velocity`` field for every pretraining task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .observation import continuous_quaternion_wxyz


@dataclass(frozen=True)
class VelocityEstimatorConfig:
    process_variance: float = 0.03059
    leg_variance: float = 0.002
    initial_variance: float = 0.1
    height_scale: float = 0.05
    vertical_velocity_scale: float = 0.35
    huber_delta: float = 0.25
    prior_temperature: float = 0.05
    innovation_gate: float = 11.34
    rejection_covariance_inflation: float = 2.0
    minimum_total_confidence: float = 0.2


class ProprioceptiveVelocityEstimator:
    """Vectorized contact-free IMU and Go2 leg-odometry Kalman filter."""

    THIGH_LENGTH = 0.213
    CALF_LENGTH = 0.213
    HIP_ABDUCTION_Y = 0.0465
    HIP_LATERAL_OFFSET = 0.0955
    # FL, FR, RL, RR order.
    HIP_X = (0.1934, 0.1934, -0.1934, -0.1934)
    LEG_SIDE = (1.0, -1.0, 1.0, -1.0)

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        dt: float = 0.02,
        config: VelocityEstimatorConfig | None = None,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.config = config or VelocityEstimatorConfig()
        if self.num_envs < 1 or self.dt <= 0:
            raise ValueError("num_envs and dt must be positive")
        for name, value in asdict(self.config).items():
            setattr(self, name, float(value))
        self.world_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        identity = torch.eye(3, device=self.device)
        self.covariance = (
            identity.expand(self.num_envs, -1, -1).clone() * self.initial_variance
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        identity = torch.eye(3, device=self.device)
        if env_ids is None:
            self.world_velocity.zero_()
            self.covariance.copy_(
                identity.expand(self.num_envs, -1, -1) * self.initial_variance
            )
        else:
            self.world_velocity[env_ids] = 0
            self.covariance[env_ids] = identity * self.initial_variance

    @staticmethod
    def rotation_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
        quaternion = continuous_quaternion_wxyz(quaternion)
        w, x, y, z = quaternion.unbind(-1)
        row0 = torch.stack(
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            -1,
        )
        row1 = torch.stack(
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            -1,
        )
        row2 = torch.stack(
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            -1,
        )
        return torch.stack((row0, row1, row2), -2)

    def _feet(self, joint_q: torch.Tensor, joint_dq: torch.Tensor):
        q = joint_q.reshape(-1, 4, 3)
        dq = joint_dq.reshape(-1, 4, 3)
        abduction, thigh, calf = q.unbind(-1)
        abduction_dq, thigh_dq, calf_dq = dq.unbind(-1)
        hip_x = q.new_tensor(self.HIP_X)
        side = q.new_tensor(self.LEG_SIDE)
        lateral = side * self.HIP_LATERAL_OFFSET
        total = thigh + calf
        x = hip_x - self.THIGH_LENGTH * thigh.sin() - self.CALF_LENGTH * total.sin()
        z_plane = -self.THIGH_LENGTH * thigh.cos() - self.CALF_LENGTH * total.cos()
        y = (
            side * self.HIP_ABDUCTION_Y
            + lateral * abduction.cos()
            - z_plane * abduction.sin()
        )
        z = lateral * abduction.sin() + z_plane * abduction.cos()
        position = torch.stack((x, y, z), -1)
        dx = (
            (-self.THIGH_LENGTH * thigh.cos() - self.CALF_LENGTH * total.cos())
            * thigh_dq
            - self.CALF_LENGTH * total.cos() * calf_dq
        )
        dz_plane = (
            (self.THIGH_LENGTH * thigh.sin() + self.CALF_LENGTH * total.sin())
            * thigh_dq
            + self.CALF_LENGTH * total.sin() * calf_dq
        )
        dy = (
            (-lateral * abduction.sin() - z_plane * abduction.cos())
            * abduction_dq
            - dz_plane * abduction.sin()
        )
        dz = (
            (lateral * abduction.cos() - z_plane * abduction.sin())
            * abduction_dq
            + dz_plane * abduction.cos()
        )
        return position, torch.stack((dx, dy, dz), -1)

    @torch.no_grad()
    def update(
        self,
        joint_q: torch.Tensor,
        joint_dq: torch.Tensor,
        angular_velocity: torch.Tensor,
        quaternion: torch.Tensor,
        accelerometer: torch.Tensor,
    ) -> torch.Tensor:
        expected = ((joint_q, 12), (joint_dq, 12), (angular_velocity, 3),
                    (quaternion, 4), (accelerometer, 3))
        for value, width in expected:
            if value.shape != (self.num_envs, width) or not torch.isfinite(value).all():
                raise ValueError("invalid proprioceptive velocity-estimator input")
        rotation = self.rotation_matrix_wxyz(quaternion)
        acceleration_world = torch.bmm(
            rotation, accelerometer.unsqueeze(-1)
        ).squeeze(-1)
        acceleration_world = acceleration_world + joint_q.new_tensor((0.0, 0.0, -9.81))
        self.world_velocity.add_(acceleration_world * self.dt)
        identity = torch.eye(3, dtype=joint_q.dtype, device=joint_q.device)
        self.covariance.add_(identity * self.process_variance * self.dt * self.dt)

        positions, foot_velocity = self._feet(joint_q, joint_dq)
        rotational = torch.linalg.cross(
            angular_velocity[:, None, :].expand_as(positions), positions, dim=-1
        )
        candidates = -(foot_velocity + rotational)
        height_delta = positions[..., 2] - positions[..., 2].amin(1, keepdim=True)
        confidence = torch.exp(
            -0.5 * (height_delta / self.height_scale).square()
            - 0.5 * (foot_velocity[..., 2] / self.vertical_velocity_scale).square()
        )
        predicted_body = torch.bmm(
            rotation.transpose(1, 2), self.world_velocity.unsqueeze(-1)
        ).squeeze(-1)
        residual_norm = torch.linalg.vector_norm(
            candidates - predicted_body[:, None, :], dim=-1
        )
        huber = torch.where(
            residual_norm > self.huber_delta,
            self.huber_delta / residual_norm.clamp_min(1e-12),
            torch.ones_like(residual_norm),
        )
        prior = torch.exp(
            -(residual_norm - residual_norm.amin(1, keepdim=True))
            / self.prior_temperature
        )
        weights = confidence.sqrt().sqrt() * huber * prior
        weight_sum = weights.sum(1, keepdim=True)
        observed_body = (candidates * weights[..., None]).sum(1) / weight_sum.clamp_min(1e-12)
        residual = candidates - observed_body[:, None, :]
        spread = (residual.square() * weights[..., None]).sum(1) / weight_sum.clamp_min(1e-12)
        effective_count = weight_sum.squeeze(-1).square() / weights.square().sum(1).clamp_min(1e-12)
        measurement_diag = self.leg_variance / effective_count.clamp_min(1.0)[:, None] + spread
        measurement_body = torch.diag_embed(measurement_diag)
        measurement_world = torch.bmm(
            torch.bmm(rotation, measurement_body), rotation.transpose(1, 2)
        )
        observed_world = torch.bmm(rotation, observed_body.unsqueeze(-1)).squeeze(-1)
        innovation = observed_world - self.world_velocity
        innovation_covariance = self.covariance + measurement_world
        solved = torch.linalg.solve(innovation_covariance, innovation.unsqueeze(-1)).squeeze(-1)
        innovation_squared = (innovation * solved).sum(-1)
        usable_measurement = (
            (confidence.sum(1) >= self.minimum_total_confidence)
            & (weight_sum.squeeze(-1) > torch.finfo(weight_sum.dtype).eps)
        )
        finite_innovation = torch.isfinite(innovation_squared)
        accepted = (
            usable_measurement
            & finite_innovation
            & (innovation_squared <= self.innovation_gate)
        )
        rejected = (
            usable_measurement
            & finite_innovation
            & (innovation_squared > self.innovation_gate)
        )
        gain = torch.linalg.solve(
            innovation_covariance.transpose(1, 2), self.covariance.transpose(1, 2)
        ).transpose(1, 2)
        updated_velocity = self.world_velocity + torch.bmm(
            gain, innovation.unsqueeze(-1)
        ).squeeze(-1)
        identity_minus_gain = identity[None] - gain
        updated_covariance = torch.bmm(
            torch.bmm(identity_minus_gain, self.covariance),
            identity_minus_gain.transpose(1, 2),
        ) + torch.bmm(torch.bmm(gain, measurement_world), gain.transpose(1, 2))
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.transpose(1, 2))
        inflated = self.covariance * self.rejection_covariance_inflation
        maximum_diagonal = torch.diagonal(inflated, dim1=-2, dim2=-1).amax(-1)
        inflated *= torch.clamp(
            self.initial_variance / maximum_diagonal.clamp_min(1e-12), max=1.0
        )[:, None, None]
        self.world_velocity.copy_(
            torch.where(accepted[:, None], updated_velocity, self.world_velocity)
        )
        self.covariance.copy_(torch.where(
            accepted[:, None, None],
            updated_covariance,
            torch.where(rejected[:, None, None], inflated, self.covariance),
        ))
        if not torch.isfinite(self.world_velocity).all() or not torch.isfinite(self.covariance).all():
            raise FloatingPointError("velocity estimator produced non-finite state")
        return torch.bmm(
            rotation.transpose(1, 2), self.world_velocity.unsqueeze(-1)
        ).squeeze(-1)
