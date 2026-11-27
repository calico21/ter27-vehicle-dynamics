"""Stage-1 G-G-V envelope extractor driven directly by Godot headless Full14DofModel."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

LTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = LTS_DIR.parents[2]
SIM_DIR = REPO_ROOT / "src" / "mirena_sim"
CACHE_DIR = LTS_DIR / "cache"
GODOT_PROBE_PATH = LTS_DIR / "godot_ggv_probe.gd"

DEFAULT_SPEEDS = [6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0]
DEFAULT_AY_FRACS = [0.0, 0.35, 0.65, 0.85, 0.96]


@dataclass
class GGVMap:
    """Cached 3D G-G-V performance envelope (v, |ay|/ay_max -> ay_max, ax_max, ax_min, p_dc)."""

    label: str
    config: Dict[str, Any]
    speeds: np.ndarray
    ay_fracs: np.ndarray
    ay_max: np.ndarray
    ax_max: np.ndarray  # shape (nv, nf), positive m/s^2
    ax_min: np.ndarray  # shape (nv, nf), negative m/s^2
    p_acc_kw: np.ndarray
    p_brk_kw: np.ndarray
    mass_kg: float
    cl_a: float
    cd_a: float
    use_3d_kinematics: bool

    def __post_init__(self) -> None:
        # Ensure grid completeness up to ay_frac = 1.0 (zero extra traction reserve at peak lateral)
        if self.ay_fracs[-1] < 0.999:
            self.ay_fracs = np.append(self.ay_fracs, 1.0)
            # At ay_frac = 1.0 (pure limit cornering), available positive ax drops smoothly to <= 0.15 * ax_max(0)
            zero_pad_acc = np.minimum(0.15 * self.ax_max[:, -1:], 0.25)
            # Deceleration at limit cornering still includes drag + subtle trail braking capacity
            zero_pad_brk = 0.25 * self.ax_min[:, -1:]
            self.ax_max = np.hstack([self.ax_max, zero_pad_acc])
            self.ax_min = np.hstack([self.ax_min, zero_pad_brk])
            self.p_acc_kw = np.hstack([self.p_acc_kw, self.p_acc_kw[:, -1:]])
            self.p_brk_kw = np.hstack([self.p_brk_kw, 0.25 * self.p_brk_kw[:, -1:]])

        self._itp_ax_max = RegularGridInterpolator(
            (self.speeds, self.ay_fracs), self.ax_max, bounds_error=False, fill_value=None
        )
        self._itp_ax_min = RegularGridInterpolator(
            (self.speeds, self.ay_fracs), self.ax_min, bounds_error=False, fill_value=None
        )
        self._itp_p_acc = RegularGridInterpolator(
            (self.speeds, self.ay_fracs), self.p_acc_kw, bounds_error=False, fill_value=None
        )
        self._itp_p_brk = RegularGridInterpolator(
            (self.speeds, self.ay_fracs), self.p_brk_kw, bounds_error=False, fill_value=None
        )

    def max_ay(self, v_ms: float | np.ndarray) -> float | np.ndarray:
        vq = np.clip(v_ms, float(self.speeds[0]), float(self.speeds[-1]))
        # Above 30 m/s, preserve top-speed aero lateral capability gently clamped
        out = np.interp(vq, self.speeds, self.ay_max)
        if np.ndim(v_ms) == 0:
            return float(out)
        return out

    def max_ax(self, v_ms: float, ay_abs_ms2: float) -> float:
        ay_lim = max(self.max_ay(v_ms), 0.5)
        frac = float(np.clip(abs(ay_abs_ms2) / ay_lim, 0.0, 1.0))
        vq = float(np.clip(v_ms, self.speeds[0], 38.0))
        if vq > float(self.speeds[-1]):
            # Power-limited extrapolation above 30 m/s using measured power at 30 m/s minus extra drag
            p30_kw = float(self._itp_p_acc(np.array([[float(self.speeds[-1]), frac]]))[0])
            ax30 = float(self._itp_ax_max(np.array([[float(self.speeds[-1]), frac]]))[0])
            df_drag = 0.5 * 1.225 * self.cd_a * (vq * vq - float(self.speeds[-1]) ** 2)
            # Wheel thrust drops roughly proportionally to P_wheel/v
            f30 = max(p30_kw * 900.0, 0.0) / max(float(self.speeds[-1]), 1.0)
            fv = max(p30_kw * 900.0, 0.0) / max(vq, 1.0)
            ax_ext = ax30 + (fv - f30 - df_drag) / max(self.mass_kg, 100.0)
            return float(max(ax_ext, -2.0))
        return float(self._itp_ax_max(np.array([[vq, frac]]))[0])

    def min_ax(self, v_ms: float, ay_abs_ms2: float) -> float:
        ay_lim = max(self.max_ay(v_ms), 0.5)
        frac = float(np.clip(abs(ay_abs_ms2) / ay_lim, 0.0, 1.0))
        vq = float(np.clip(v_ms, self.speeds[0], float(self.speeds[-1])))
        val = float(self._itp_ax_min(np.array([[vq, frac]]))[0])
        if v_ms > float(self.speeds[-1]):
            df_drag = 0.5 * 1.225 * self.cd_a * (v_ms * v_ms - float(self.speeds[-1]) ** 2)
            val -= df_drag / max(self.mass_kg, 100.0)
        return float(min(val, -0.5))

    def est_power_kw(self, v_ms: float, ax_ms2: float, ay_abs_ms2: float) -> float:
        ay_lim = max(self.max_ay(v_ms), 0.5)
        frac = float(np.clip(abs(ay_abs_ms2) / ay_lim, 0.0, 1.0))
        vq = float(np.clip(v_ms, self.speeds[0], float(self.speeds[-1])))
        if ax_ms2 >= 0.0:
            ax_top = max(self.max_ax(v_ms, ay_abs_ms2), 0.2)
            ratio = float(np.clip(ax_ms2 / ax_top, 0.0, 1.0))
            p_max = float(self._itp_p_acc(np.array([[vq, frac]]))[0])
            # Road-load cruise baseline ~ (drag + rolling resistance) * v / eta
            p_cruise = (0.5 * 1.225 * self.cd_a * v_ms * v_ms + 0.015 * self.mass_kg * 9.81) * v_ms / 900.0
            return float(max(p_cruise, ratio * p_max))
        else:
            ax_bot = min(self.min_ax(v_ms, ay_abs_ms2), -0.5)
            ratio = float(np.clip(abs(ax_ms2) / abs(ax_bot), 0.0, 1.0))
            p_reg = float(self._itp_p_brk(np.array([[vq, frac]]))[0])
            # Mild decel up to ~0.35g captures up to full regen curve; heavier decel saturates regen near -25 kW
            return float(np.clip(ratio * 1.4 * p_reg, p_reg, 0.0))

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            label=self.label,
            config_json=json.dumps(self.config),
            speeds=self.speeds,
            ay_fracs=self.ay_fracs,
            ay_max=self.ay_max,
            ax_max=self.ax_max,
            ax_min=self.ax_min,
            p_acc_kw=self.p_acc_kw,
            p_brk_kw=self.p_brk_kw,
            mass_kg=self.mass_kg,
            cl_a=self.cl_a,
            cd_a=self.cd_a,
            use_3d_kinematics=int(self.use_3d_kinematics),
        )

    @classmethod
    def load_npz(cls, path: Path) -> "GGVMap":
        d = np.load(path, allow_pickle=False)
        return cls(
            label=str(d["label"].item()),
            config=json.loads(str(d["config_json"].item())),
            speeds=np.asarray(d["speeds"], dtype=np.float64),
            ay_fracs=np.asarray(d["ay_fracs"], dtype=np.float64),
            ay_max=np.asarray(d["ay_max"], dtype=np.float64),
            ax_max=np.asarray(d["ax_max"], dtype=np.float64),
            ax_min=np.asarray(d["ax_min"], dtype=np.float64),
            p_acc_kw=np.asarray(d["p_acc_kw"], dtype=np.float64),
            p_brk_kw=np.asarray(d["p_brk_kw"], dtype=np.float64),
            mass_kg=float(d["mass_kg"].item()),
            cl_a=float(d["cl_a"].item()),
            cd_a=float(d["cd_a"].item()),
            use_3d_kinematics=bool(int(d["use_3d_kinematics"].item())),
        )


def _config_cache_key(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _enforce_physical_monotonicity(
    ay_max: np.ndarray,
    ax_max: np.ndarray,
    ax_min: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure smooth, physically monotonic envelopes (friction-ellipse decay with ay, downforce growth with v)."""
    # 1. Enforce non-decreasing ay_max with speed (steering-lock + aero downforce both strictly grow with v)
    ay_m = np.maximum.accumulate(ay_max.copy())
    # 2. Enforce coupled traction/braking decay as lateral utilization rises
    ax_p = ax_max.copy()
    ax_n = ax_min.copy()
    for iv in range(ax_p.shape[0]):
        for jf in range(1, ax_p.shape[1]):
            ax_p[iv, jf] = min(ax_p[iv, jf], ax_p[iv, jf - 1])
            ax_n[iv, jf] = max(ax_n[iv, jf], ax_n[iv, jf - 1])  # closer to 0 (less negative)
        ax_p[iv] = np.maximum(ax_p[iv], 0.0)
        ax_n[iv] = np.minimum(ax_n[iv], -0.5)
    return ay_m, ax_p, ax_n


def extract_or_load_ggv(
    label: str = "tv4_auto",
    ecu_driving_mode: int = 2,
    tv_4wd_mode: int = 0,
    use_3d_kinematics: bool = True,
    mass_kg: Optional[float] = None,
    cl_a_scale: Optional[float] = None,
    p_eff_max_w: Optional[float] = None,
    mech_brake_start: Optional[float] = None,
    speeds: Optional[List[float]] = None,
    force_recompute: bool = False,
    grip_profile: str = "ttc_lab",
) -> GGVMap:
    """Extract G-G-V envelope via Godot Full14DofModel or load cached NPZ."""
    cfg: Dict[str, Any] = {
        "ecu_driving_mode": int(ecu_driving_mode),
        "tv_4wd_mode": int(tv_4wd_mode),
        "use_3d_kinematics": bool(use_3d_kinematics),
        "speeds_ms": list(speeds or DEFAULT_SPEEDS),
        "ay_fracs": list(DEFAULT_AY_FRACS),
    }
    if mass_kg is not None:
        cfg["mass_kg"] = float(mass_kg)
    if cl_a_scale is not None:
        cfg["cl_a_scale"] = float(cl_a_scale)
    if p_eff_max_w is not None:
        cfg["p_eff_max_w"] = float(p_eff_max_w)
    if mech_brake_start is not None:
        cfg["mech_brake_start"] = float(mech_brake_start)
    if grip_profile and grip_profile != "ttc_lab":
        cfg["grip_profile"] = str(grip_profile)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _config_cache_key(cfg)
    npz_path = CACHE_DIR / f"ggv_{label}_{key}.npz"
    json_path = CACHE_DIR / f"ggv_{label}_{key}.json"

    if not force_recompute and npz_path.exists():
        return GGVMap.load_npz(npz_path)

    with tempfile.TemporaryDirectory(prefix="mirena_ggv_") as tmp:
        tmp_dir = Path(tmp)
        cfg_file = tmp_dir / "ggv_cfg.json"
        out_file = tmp_dir / "ggv_raw.json"
        cfg_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        cmd = [
            "godot",
            "--headless",
            "--path",
            str(SIM_DIR),
            "-s",
            str(GODOT_PROBE_PATH),
            "--",
            str(cfg_file),
            str(out_file),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out_file.exists():
            raise RuntimeError(
                f"Godot GGV probe failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        raw = json.loads(out_file.read_text(encoding="utf-8"))

    rows = raw["rows"]
    sp_arr = np.array([float(r["v_ms"]) for r in rows], dtype=np.float64)
    af_arr = np.array(raw["ay_fracs"], dtype=np.float64)
    ay_arr = np.array([float(r["ay_max"]) for r in rows], dtype=np.float64)
    ax_max_arr = np.array([r["ax_max"] for r in rows], dtype=np.float64)
    ax_min_arr = np.array([r["ax_min"] for r in rows], dtype=np.float64)
    p_acc_arr = np.array([r["p_acc_kw"] for r in rows], dtype=np.float64)
    p_brk_arr = np.array([r["p_brk_kw"] for r in rows], dtype=np.float64)

    ay_arr, ax_max_arr, ax_min_arr = _enforce_physical_monotonicity(ay_arr, ax_max_arr, ax_min_arr)

    ggv = GGVMap(
        label=label,
        config=cfg,
        speeds=sp_arr,
        ay_fracs=af_arr,
        ay_max=ay_arr,
        ax_max=ax_max_arr,
        ax_min=ax_min_arr,
        p_acc_kw=p_acc_arr,
        p_brk_kw=p_brk_arr,
        mass_kg=float(raw["mass_kg"]),
        cl_a=float(raw["cl_a"]),
        cd_a=float(raw["cd_a"]),
        use_3d_kinematics=bool(raw["use_3d_kinematics"]),
    )
    ggv.save_npz(npz_path)
    json_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return ggv
