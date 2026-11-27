#!/usr/bin/env python3
"""
Sensor Realism & CAN Latency Profile (SENSOR_IDEAL vs. SENSOR_TRACK_REALISM).

Decoupled, deterministic sensor degradation and kinematic-inertial observer
module with 1-to-1 numerical parity against Godot 4 SensorRealismFilter.gd.

Features in SENSOR_TRACK_REALISM:
- Differentiated CAN transport latency ring-buffers:
  * Accelerometer channel (ax, ay): default 125.0 ms (TeR26 CAN audit)
  * Steering & bus channels (delta, wz, omega_4): default 20.0 ms
- Discrete 200 Hz zero-order sample-and-hold and ADC/encoder quantization
  (including 0.05 rad/s wheel speed encoder quantization).
- Deterministic Gaussian IMU/steering/encoder noise (scaled by imu_noise_scale)
  plus residual uncancelled IMU bias (ax_bias, ay_bias, wz_bias).
- Kinematic-inertial EKF estimating beta_hat and load-transfer observer
  estimating fz_hat_4 strictly from delayed/noisy sensor measurements.
"""

from __future__ import annotations

from enum import IntEnum
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np


class SensorMode(IntEnum):
    SENSOR_IDEAL = 0
    SENSOR_TRACK_REALISM = 1


_DEFAULT_CONFIG: Dict[str, Any] = {
    "default_mode": "SENSOR_IDEAL",
    "seed": 20260427,
    "sample_rate_hz": 200.0,
    "can_accel_latency_ms": 125.0,
    "can_steer_latency_ms": 20.0,
    "imu_noise_scale": 1.0,
    "ax_noise_std_ms2": 0.18,
    "ay_noise_std_ms2": 0.18,
    "wz_noise_std_rads": 0.012,
    "delta_noise_std_rad": 0.0015,
    "omega_noise_std_rads": 0.08,
    "ax_bias_ms2": 0.08,
    "ay_bias_ms2": -0.06,
    "wz_bias_rads": 0.003,
    "accel_quant_ms2": 0.01,
    "gyro_quant_rads": 0.001,
    "steer_quant_rad": 0.0005,
    "wheel_omega_quant_rads": 0.05,
    "mass_kg": 290.0,
    "wheelbase_m": 1.54,
    "front_weight_dist": 0.45,
    "track_front_m": 1.23,
    "track_rear_m": 1.23,
    "h_cog_m": 0.29,
    "lltd_front": 0.455,
    "cl_a_m2": 6.0,
    "aero_balance_front": 0.48,
    "wheel_radius_m": 0.2032,
    "tau_wt_s": 0.035,
    "gravity_ms2": 9.81,
    "rho_air_kgm3": 1.225,
    "ekf_q_vx": 0.15,
    "ekf_q_vy": 0.35,
    "ekf_r_vx": 0.05,
    "ekf_r_vy_base": 0.25,
    "ekf_r_vy_ay_gain": 0.04,
    "beta_clamp_rad": 0.5235987756,
    "fz_max_n": 4000.0,
}


def _clamp(val: float, lo: float, hi: float) -> float:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def _quantize(val: float, step: float) -> float:
    if step <= 0.0:
        return val
    return math.floor(val / step + 0.5) * step


def _median4(v0: float, v1: float, v2: float, v3: float) -> float:
    arr = [v0, v1, v2, v3]
    arr.sort()
    return 0.5 * (arr[1] + arr[2])


def _parse_mode(mode: Union[int, str, SensorMode]) -> SensorMode:
    if isinstance(mode, SensorMode):
        return mode
    if isinstance(mode, int):
        return SensorMode.SENSOR_TRACK_REALISM if mode == 1 else SensorMode.SENSOR_IDEAL
    if isinstance(mode, str):
        key = mode.strip().upper()
        if key in ("1", "SENSOR_TRACK_REALISM", "TRACK_REALISM", "REALISM", "TRACK"):
            return SensorMode.SENSOR_TRACK_REALISM
        return SensorMode.SENSOR_IDEAL
    return SensorMode.SENSOR_IDEAL


class _RingBufferScalar:
    """Fixed-size ring buffer with exact integer-tick transport delay."""

    __slots__ = ("delay_ticks", "buf", "head")

    def __init__(self, delay_ticks: int) -> None:
        self.delay_ticks: int = max(0, int(delay_ticks))
        self.buf: List[float] = [0.0] * (self.delay_ticks + 1)
        self.head: int = 0

    def fill(self, value: float) -> None:
        v = float(value)
        for i in range(len(self.buf)):
            self.buf[i] = v
        self.head = 0

    def push_pop(self, value: float) -> float:
        n = len(self.buf)
        self.buf[self.head] = float(value)
        read_idx = (self.head + 1) % n
        out = self.buf[read_idx]
        self.head = read_idx
        return out


class SensorRealismProfile:
    """Deterministic sensor realism filter & kinematic-inertial EKF observer."""

    def __init__(
        self,
        mode: Union[int, str, SensorMode] = SensorMode.SENSOR_IDEAL,
        seed: int = 20260427,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.mode: SensorMode = _parse_mode(mode)
        self.seed: int = int(seed)

        # Sampling & CAN transport latency
        self.sample_rate_hz: float = float(_DEFAULT_CONFIG["sample_rate_hz"])
        self.can_accel_latency_ms: float = float(_DEFAULT_CONFIG["can_accel_latency_ms"])
        self.can_steer_latency_ms: float = float(_DEFAULT_CONFIG["can_steer_latency_ms"])

        # Noise & residual bias
        self.imu_noise_scale: float = float(_DEFAULT_CONFIG["imu_noise_scale"])
        self.ax_noise_std_ms2: float = float(_DEFAULT_CONFIG["ax_noise_std_ms2"])
        self.ay_noise_std_ms2: float = float(_DEFAULT_CONFIG["ay_noise_std_ms2"])
        self.wz_noise_std_rads: float = float(_DEFAULT_CONFIG["wz_noise_std_rads"])
        self.delta_noise_std_rad: float = float(_DEFAULT_CONFIG["delta_noise_std_rad"])
        self.omega_noise_std_rads: float = float(_DEFAULT_CONFIG["omega_noise_std_rads"])
        self.ax_bias_ms2: float = float(_DEFAULT_CONFIG["ax_bias_ms2"])
        self.ay_bias_ms2: float = float(_DEFAULT_CONFIG["ay_bias_ms2"])
        self.wz_bias_rads: float = float(_DEFAULT_CONFIG["wz_bias_rads"])

        # ADC & encoder quantization
        self.accel_quant_ms2: float = float(_DEFAULT_CONFIG["accel_quant_ms2"])
        self.gyro_quant_rads: float = float(_DEFAULT_CONFIG["gyro_quant_rads"])
        self.steer_quant_rad: float = float(_DEFAULT_CONFIG["steer_quant_rad"])
        self.wheel_omega_quant_rads: float = float(_DEFAULT_CONFIG["wheel_omega_quant_rads"])

        # Vehicle physical parameters for EKF & Fz observer
        self.mass_kg: float = float(_DEFAULT_CONFIG["mass_kg"])
        self.wheelbase_m: float = float(_DEFAULT_CONFIG["wheelbase_m"])
        self.front_weight_dist: float = float(_DEFAULT_CONFIG["front_weight_dist"])
        self.track_front_m: float = float(_DEFAULT_CONFIG["track_front_m"])
        self.track_rear_m: float = float(_DEFAULT_CONFIG["track_rear_m"])
        self.h_cog_m: float = float(_DEFAULT_CONFIG["h_cog_m"])
        self.lltd_front: float = float(_DEFAULT_CONFIG["lltd_front"])
        self.cl_a_m2: float = float(_DEFAULT_CONFIG["cl_a_m2"])
        self.aero_balance_front: float = float(_DEFAULT_CONFIG["aero_balance_front"])
        self.wheel_radius_m: float = float(_DEFAULT_CONFIG["wheel_radius_m"])
        self.tau_wt_s: float = float(_DEFAULT_CONFIG["tau_wt_s"])
        self.gravity_ms2: float = float(_DEFAULT_CONFIG["gravity_ms2"])
        self.rho_air_kgm3: float = float(_DEFAULT_CONFIG["rho_air_kgm3"])

        # EKF gains & bounds
        self.ekf_q_vx: float = float(_DEFAULT_CONFIG["ekf_q_vx"])
        self.ekf_q_vy: float = float(_DEFAULT_CONFIG["ekf_q_vy"])
        self.ekf_r_vx: float = float(_DEFAULT_CONFIG["ekf_r_vx"])
        self.ekf_r_vy_base: float = float(_DEFAULT_CONFIG["ekf_r_vy_base"])
        self.ekf_r_vy_ay_gain: float = float(_DEFAULT_CONFIG["ekf_r_vy_ay_gain"])
        self.beta_clamp_rad: float = float(_DEFAULT_CONFIG["beta_clamp_rad"])
        self.fz_max_n: float = float(_DEFAULT_CONFIG["fz_max_n"])

        # Internal state
        self._rng: np.random.Generator = np.random.default_rng(self.seed)
        self._initialized: bool = False
        self._sample_accum_s: float = 0.0
        self._tick_count: int = 0
        self._ekf_vx: float = 0.0
        self._ekf_vy: float = 0.0
        self._ekf_p_vx: float = 0.2
        self._ekf_p_vy: float = 0.2
        self._ax_wt: float = 0.0
        self._ay_wt: float = 0.0
        self._beta_hat: float = 0.0
        self._fz_hat_4: List[float] = [640.0, 640.0, 780.0, 780.0]

        self._buf_ax: _RingBufferScalar = _RingBufferScalar(0)
        self._buf_ay: _RingBufferScalar = _RingBufferScalar(0)
        self._buf_wz: _RingBufferScalar = _RingBufferScalar(0)
        self._buf_delta: _RingBufferScalar = _RingBufferScalar(0)
        self._buf_omega: List[_RingBufferScalar] = [_RingBufferScalar(0) for _ in range(4)]

        self._last_output: Dict[str, Any] = {}

        if config is not None:
            self.configure(**config)
        else:
            self.reset(seed=self.seed)

    @classmethod
    def from_tv_config_json(
        cls,
        json_path: Union[str, Path],
        mode: Optional[Union[int, str, SensorMode]] = None,
    ) -> "SensorRealismProfile":
        path = Path(json_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        sr_cfg: Dict[str, Any] = dict(data.get("sensor_realism", {}))
        phys: Dict[str, Any] = dict(data.get("physical", {}))
        merged: Dict[str, Any] = {}
        for k in (
            "mass_kg",
            "wheelbase_m",
            "front_weight_dist",
            "track_front_m",
            "track_rear_m",
            "h_cog_m",
            "lltd_front",
            "cl_a_m2",
            "aero_balance_front",
            "tau_wt_s",
        ):
            if k in phys:
                merged[k] = phys[k]
        if "wheel_radius_eff_m" in phys:
            merged["wheel_radius_m"] = phys["wheel_radius_eff_m"]
        merged.update(sr_cfg)
        init_mode = mode if mode is not None else merged.pop("default_mode", SensorMode.SENSOR_IDEAL)
        merged.pop("default_mode", None)
        seed = int(merged.pop("seed", _DEFAULT_CONFIG["seed"]))
        return cls(mode=init_mode, seed=seed, config=merged)

    def set_mode(self, mode: Union[int, str, SensorMode]) -> None:
        new_mode = _parse_mode(mode)
        if new_mode != self.mode:
            self.mode = new_mode
            self.reset()

    def configure(self, config_dict: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        params: Dict[str, Any] = {}
        if config_dict is not None:
            params.update(config_dict)
        params.update(kwargs)

        # Support convenient aliases
        if "ax_bias" in params:
            params.setdefault("ax_bias_ms2", params["ax_bias"])
        if "ay_bias" in params:
            params.setdefault("ay_bias_ms2", params["ay_bias"])
        if "wz_bias" in params:
            params.setdefault("wz_bias_rads", params["wz_bias"])
        if "can_latency_ms" in params:
            params.setdefault("can_accel_latency_ms", params["can_latency_ms"])

        rebuild_buffers = False
        for key in (
            "sample_rate_hz",
            "can_accel_latency_ms",
            "can_steer_latency_ms",
            "imu_noise_scale",
            "ax_noise_std_ms2",
            "ay_noise_std_ms2",
            "wz_noise_std_rads",
            "delta_noise_std_rad",
            "omega_noise_std_rads",
            "ax_bias_ms2",
            "ay_bias_ms2",
            "wz_bias_rads",
            "accel_quant_ms2",
            "gyro_quant_rads",
            "steer_quant_rad",
            "wheel_omega_quant_rads",
            "mass_kg",
            "wheelbase_m",
            "front_weight_dist",
            "track_front_m",
            "track_rear_m",
            "h_cog_m",
            "lltd_front",
            "cl_a_m2",
            "aero_balance_front",
            "wheel_radius_m",
            "tau_wt_s",
            "gravity_ms2",
            "rho_air_kgm3",
            "ekf_q_vx",
            "ekf_q_vy",
            "ekf_r_vx",
            "ekf_r_vy_base",
            "ekf_r_vy_ay_gain",
            "beta_clamp_rad",
            "fz_max_n",
        ):
            if key in params:
                val = float(params[key])
                if key in ("sample_rate_hz", "can_accel_latency_ms", "can_steer_latency_ms"):
                    if key == "sample_rate_hz":
                        val = max(val, 10.0)
                    else:
                        val = max(val, 0.0)
                    if abs(getattr(self, key) - val) > 1e-9:
                        rebuild_buffers = True
                elif key == "imu_noise_scale":
                    val = max(val, 0.0)
                setattr(self, key, val)

        if "mode" in params:
            self.mode = _parse_mode(params["mode"])
        if "seed" in params:
            self.seed = int(params["seed"])
            rebuild_buffers = True

        if rebuild_buffers or not self._buf_ax.buf:
            self.reset(seed=self.seed)

    def _compute_delay_ticks(self, latency_ms: float) -> int:
        return max(0, int(math.floor((max(latency_ms, 0.0) / 1000.0) * self.sample_rate_hz + 0.5)))

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._initialized = False
        self._sample_accum_s = 0.0
        self._tick_count = 0

        accel_ticks = self._compute_delay_ticks(self.can_accel_latency_ms)
        steer_ticks = self._compute_delay_ticks(self.can_steer_latency_ms)
        self._buf_ax = _RingBufferScalar(accel_ticks)
        self._buf_ay = _RingBufferScalar(accel_ticks)
        self._buf_wz = _RingBufferScalar(steer_ticks)
        self._buf_delta = _RingBufferScalar(steer_ticks)
        self._buf_omega = [_RingBufferScalar(steer_ticks) for _ in range(4)]

        self._ekf_vx = 0.0
        self._ekf_vy = 0.0
        self._ekf_p_vx = 0.2
        self._ekf_p_vy = 0.2
        self._ax_wt = 0.0
        self._ay_wt = 0.0
        self._beta_hat = 0.0
        fz_f = 0.5 * self.mass_kg * self.gravity_ms2 * self.front_weight_dist
        fz_r = 0.5 * self.mass_kg * self.gravity_ms2 * (1.0 - self.front_weight_dist)
        self._fz_hat_4 = [fz_f, fz_f, fz_r, fz_r]
        self._last_output = {}

    def _randn(self) -> float:
        return float(self._rng.standard_normal())

    def _step_ekf_and_fz(
        self,
        ax_m: float,
        ay_m: float,
        wz_m: float,
        delta_m: float,
        omega_m: List[float],
        dt_tick: float,
        is_init: bool,
    ) -> None:
        vx_wheel = _median4(
            omega_m[0] * self.wheel_radius_m,
            omega_m[1] * self.wheel_radius_m,
            omega_m[2] * self.wheel_radius_m,
            omega_m[3] * self.wheel_radius_m,
        )

        if is_init:
            self._ekf_vx = vx_wheel
            self._ekf_vy = 0.0
            self._ekf_p_vx = 0.2
            self._ekf_p_vy = 0.2
            self._ax_wt = ax_m
            self._ay_wt = ay_m
        else:
            vx_pred = self._ekf_vx + (ax_m + wz_m * self._ekf_vy) * dt_tick
            vy_pred = self._ekf_vy + (ay_m - wz_m * self._ekf_vx) * dt_tick
            self._ekf_p_vx += self.ekf_q_vx * dt_tick
            self._ekf_p_vy += self.ekf_q_vy * dt_tick

            k_vx = self._ekf_p_vx / (self._ekf_p_vx + self.ekf_r_vx)
            self._ekf_vx = vx_pred + k_vx * (vx_wheel - vx_pred)
            self._ekf_p_vx = (1.0 - k_vx) * self._ekf_p_vx

            if abs(self._ekf_vx) > 0.5:
                vy_kin = self._ekf_vx * self.front_weight_dist * math.tan(_clamp(delta_m, -0.6, 0.6))
            else:
                vy_kin = 0.0
            r_vy = self.ekf_r_vy_base + self.ekf_r_vy_ay_gain * ay_m * ay_m
            k_vy = self._ekf_p_vy / (self._ekf_p_vy + r_vy)
            self._ekf_vy = vy_pred + k_vy * (vy_kin - vy_pred)
            self._ekf_p_vy = (1.0 - k_vy) * self._ekf_p_vy
            if abs(self._ekf_vx) < 0.5:
                self._ekf_vy *= 0.8

        if abs(self._ekf_vx) > 0.2:
            self._beta_hat = _clamp(
                math.atan2(self._ekf_vy, max(abs(self._ekf_vx), 0.5)),
                -self.beta_clamp_rad,
                self.beta_clamp_rad,
            )
        else:
            self._beta_hat = 0.0

        alpha_wt = 1.0 - math.exp(-dt_tick / max(self.tau_wt_s, 1e-4))
        self._ax_wt += alpha_wt * (ax_m - self._ax_wt)
        self._ay_wt += alpha_wt * (ay_m - self._ay_wt)

        f_aero = 0.5 * self.rho_air_kgm3 * self.cl_a_m2 * self._ekf_vx * self._ekf_vx
        fz_f_base = self.mass_kg * self.gravity_ms2 * self.front_weight_dist + f_aero * self.aero_balance_front
        fz_r_base = self.mass_kg * self.gravity_ms2 * (1.0 - self.front_weight_dist) + f_aero * (1.0 - self.aero_balance_front)
        dfz_long = self.mass_kg * self._ax_wt * self.h_cog_m / max(self.wheelbase_m, 0.1)
        fz_f = fz_f_base - dfz_long
        fz_r = fz_r_base + dfz_long
        dfz_lat_f = self.lltd_front * self.mass_kg * self._ay_wt * self.h_cog_m / max(self.track_front_m, 0.1)
        dfz_lat_r = (1.0 - self.lltd_front) * self.mass_kg * self._ay_wt * self.h_cog_m / max(self.track_rear_m, 0.1)

        self._fz_hat_4 = [
            _clamp(0.5 * fz_f - dfz_lat_f, 0.0, self.fz_max_n),
            _clamp(0.5 * fz_f + dfz_lat_f, 0.0, self.fz_max_n),
            _clamp(0.5 * fz_r - dfz_lat_r, 0.0, self.fz_max_n),
            _clamp(0.5 * fz_r + dfz_lat_r, 0.0, self.fz_max_n),
        ]

    def _run_200hz_tick(
        self,
        ax: float,
        ay: float,
        wz: float,
        delta: float,
        omega_4: List[float],
        dt_tick: float,
    ) -> Dict[str, Any]:
        is_init = not self._initialized
        if is_init:
            self._buf_ax.fill(ax)
            self._buf_ay.fill(ay)
            self._buf_wz.fill(wz)
            self._buf_delta.fill(delta)
            for i in range(4):
                self._buf_omega[i].fill(omega_4[i])
            self._initialized = True

        ax_d = self._buf_ax.push_pop(ax)
        ay_d = self._buf_ay.push_pop(ay)
        wz_d = self._buf_wz.push_pop(wz)
        delta_d = self._buf_delta.push_pop(delta)
        omega_d = [self._buf_omega[i].push_pop(omega_4[i]) for i in range(4)]

        n_scale = self.imu_noise_scale
        ax_n = ax_d + self.ax_bias_ms2 + (self.ax_noise_std_ms2 * n_scale * self._randn() if n_scale > 0.0 else 0.0)
        ay_n = ay_d + self.ay_bias_ms2 + (self.ay_noise_std_ms2 * n_scale * self._randn() if n_scale > 0.0 else 0.0)
        wz_n = wz_d + self.wz_bias_rads + (self.wz_noise_std_rads * n_scale * self._randn() if n_scale > 0.0 else 0.0)
        delta_n = delta_d + (self.delta_noise_std_rad * n_scale * self._randn() if n_scale > 0.0 else 0.0)
        omega_n = [
            omega_d[i] + (self.omega_noise_std_rads * n_scale * self._randn() if n_scale > 0.0 else 0.0)
            for i in range(4)
        ]

        ax_m = _quantize(ax_n, self.accel_quant_ms2)
        ay_m = _quantize(ay_n, self.accel_quant_ms2)
        wz_m = _quantize(wz_n, self.gyro_quant_rads)
        delta_m = _quantize(delta_n, self.steer_quant_rad)
        omega_m = [_quantize(omega_n[i], self.wheel_omega_quant_rads) for i in range(4)]

        self._step_ekf_and_fz(ax_m, ay_m, wz_m, delta_m, omega_m, dt_tick, is_init)
        self._tick_count += 1

        return {
            "ax": ax_m,
            "ay": ay_m,
            "wz": wz_m,
            "vx": self._ekf_vx,
            "vy": self._ekf_vy,
            "delta": delta_m,
            "fz_4": list(self._fz_hat_4),
            "omega_4": omega_m,
            "beta_hat": self._beta_hat,
            "fz_hat_4": list(self._fz_hat_4),
            "mode": int(self.mode),
            "mode_name": self.mode.name,
            "can_latency_ms": self.can_accel_latency_ms,
            "can_steer_latency_ms": self.can_steer_latency_ms,
            "imu_noise_scale": self.imu_noise_scale,
        }

    def step(
        self,
        ax: float,
        ay: float,
        wz: float,
        vx: float,
        vy: float,
        delta: float,
        fz_4: Sequence[float],
        omega_4: Sequence[float],
        dt: float,
    ) -> Dict[str, Any]:
        fz_list = [float(fz_4[i]) for i in range(4)]
        omega_list = [float(omega_4[i]) for i in range(4)]

        if self.mode == SensorMode.SENSOR_IDEAL:
            beta_gt = math.atan2(vy, max(abs(vx), 0.5)) if abs(vx) > 0.1 else 0.0
            self._ekf_vx = float(vx)
            self._ekf_vy = float(vy)
            self._beta_hat = beta_gt
            self._fz_hat_4 = list(fz_list)
            out = {
                "ax": float(ax),
                "ay": float(ay),
                "wz": float(wz),
                "vx": float(vx),
                "vy": float(vy),
                "delta": float(delta),
                "fz_4": fz_list,
                "omega_4": omega_list,
                "beta_hat": beta_gt,
                "fz_hat_4": list(fz_list),
                "mode": int(SensorMode.SENSOR_IDEAL),
                "mode_name": SensorMode.SENSOR_IDEAL.name,
                "can_latency_ms": 0.0,
                "can_steer_latency_ms": 0.0,
                "imu_noise_scale": 0.0,
            }
            self._last_output = out
            return out

        sample_dt = 1.0 / max(self.sample_rate_hz, 10.0)
        if not self._initialized:
            self._last_output = self._run_200hz_tick(
                float(ax), float(ay), float(wz), float(delta), omega_list, sample_dt
            )
            self._sample_accum_s = 0.0
            return dict(self._last_output)

        if dt > 0.0:
            self._sample_accum_s += float(dt)
            while self._sample_accum_s >= sample_dt - 1e-9:
                self._sample_accum_s -= sample_dt
                if self._sample_accum_s < 0.0:
                    self._sample_accum_s = 0.0
                self._last_output = self._run_200hz_tick(
                    float(ax), float(ay), float(wz), float(delta), omega_list, sample_dt
                )

        return dict(self._last_output)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "mode": int(self.mode),
            "mode_name": self.mode.name,
            "seed": self.seed,
            "sample_rate_hz": self.sample_rate_hz,
            "can_accel_latency_ms": self.can_accel_latency_ms,
            "can_steer_latency_ms": self.can_steer_latency_ms,
            "accel_delay_ticks": self._buf_ax.delay_ticks,
            "steer_delay_ticks": self._buf_delta.delay_ticks,
            "imu_noise_scale": self.imu_noise_scale,
            "ekf_vx": self._ekf_vx,
            "ekf_vy": self._ekf_vy,
            "beta_hat": self._beta_hat,
            "fz_hat_4": list(self._fz_hat_4),
            "tick_count": self._tick_count,
        }
