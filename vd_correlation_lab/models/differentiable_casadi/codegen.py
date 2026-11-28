#!/usr/bin/env python3
"""
CasADi C-Codegen compiler and loader for libter_diff_14dof.so.

Compiles the C2-smooth 14-DOF CasADi SX functions:
  - step_dynamics(x, u, theta, w) -> (x_next, outputs)
  - step_jacobians(x, u, theta, w) -> (A_x, B_u, S_theta, G_w)
  - step_dynamics_dt(x, u, theta, w, dt) -> (x_next, outputs)
  - step_jacobians_dt(x, u, theta, w, dt) -> (A_x, B_u, S_theta, G_w)
  - output_jacobians(x, u, theta, w) -> (C_x, D_u, O_theta, O_w)
  - f_ode(x, u, theta, w) -> (x_dot, outputs)

into outputs/casadi_cache/libter_diff_14dof.so using:
  gcc -O3 -march=native -fPIC -shared
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import casadi as ca
import numpy as np

from vd_correlation_lab.models.differentiable_casadi.vehicle_14dof_casadi import (
    DifferentiableVehicle14Dof,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "outputs" / "casadi_cache"
DEFAULT_SO_PATH = DEFAULT_CACHE_DIR / "libter_diff_14dof.so"


@dataclass
class Compiled14DofLibrary:
    """Wrapper around compiled CasADi external functions from libter_diff_14dof.so."""

    so_path: Path
    step_dynamics: ca.Function
    step_jacobians: ca.Function
    step_dynamics_dt: ca.Function
    step_jacobians_dt: ca.Function
    output_jacobians: ca.Function
    f_ode: ca.Function

    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        theta: np.ndarray,
        w: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if dt is None:
            xn, yn = self.step_dynamics(x, u, theta, w)
        else:
            xn, yn = self.step_dynamics_dt(x, u, theta, w, float(dt))
        return np.asarray(xn, dtype=np.float64).ravel(), np.asarray(yn, dtype=np.float64).ravel()

    def jacobians(
        self,
        x: np.ndarray,
        u: np.ndarray,
        theta: np.ndarray,
        w: np.ndarray,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if dt is None:
            ax, bu, st, gw = self.step_jacobians(x, u, theta, w)
        else:
            ax, bu, st, gw = self.step_jacobians_dt(x, u, theta, w, float(dt))
        return (
            np.asarray(ax, dtype=np.float64),
            np.asarray(bu, dtype=np.float64),
            np.asarray(st, dtype=np.float64),
            np.asarray(gw, dtype=np.float64),
        )


def compile_14dof_library(
    model: Optional[DifferentiableVehicle14Dof] = None,
    so_path: Path = DEFAULT_SO_PATH,
    force_rebuild: bool = False,
    opt_flags: Tuple[str, ...] = ("-O3", "-march=native"),
) -> Compiled14DofLibrary:
    """
    Generate C code via CasADi CodeGenerator and compile to a shared library (.so).
    If so_path already exists and force_rebuild is False, loads directly.
    """
    so_path = Path(so_path)
    so_path.parent.mkdir(parents=True, exist_ok=True)

    if so_path.exists() and not force_rebuild:
        return load_compiled_14dof_library(so_path)

    if model is None:
        model = DifferentiableVehicle14Dof(
            vehicle_id="ter27_4wd",
            suspension_mode="3d_double_wishbone_lut",
            use_pinn=True,
            dt=0.005,
            rk4_substeps=2,
        )

    c_name = so_path.stem + ".c"
    c_path = so_path.parent / c_name

    t0 = time.perf_counter()
    cg = ca.CodeGenerator(c_name)
    cg.add(model.fn_step)
    cg.add(model.fn_jacobians)
    cg.add(model.fn_step_dt)
    cg.add(model.fn_jacobians_dt)
    cg.add(model.fn_output_jacobians)
    cg.add(model.fn_ode)
    cg.generate(str(so_path.parent) + "/")
    t_gen = time.perf_counter() - t0

    cmd = ["gcc", *opt_flags, "-fPIC", "-shared", str(c_path), "-o", str(so_path), "-lm"]
    t1 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t_cc = time.perf_counter() - t1
    if proc.returncode != 0:
        raise RuntimeError(
            f"GCC compilation failed (rc={proc.returncode}):\n"
            f"CMD: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}"
        )

    print(
        f"[codegen] Built {so_path.name} ({so_path.stat().st_size / 1024.0:.1f} KB) "
        f"in {t_gen:.2f}s codegen + {t_cc:.2f}s gcc",
        flush=True,
    )
    return load_compiled_14dof_library(so_path)


def load_compiled_14dof_library(so_path: Path = DEFAULT_SO_PATH) -> Compiled14DofLibrary:
    """Load compiled CasADi external functions from an existing .so library."""
    so_path = Path(so_path)
    if not so_path.exists():
        raise FileNotFoundError(
            f"Compiled CasADi library not found at {so_path}. Run compile_14dof_library() first."
        )
    so_str = str(so_path)
    return Compiled14DofLibrary(
        so_path=so_path,
        step_dynamics=ca.external("step_dynamics", so_str),
        step_jacobians=ca.external("step_jacobians", so_str),
        step_dynamics_dt=ca.external("step_dynamics_dt", so_str),
        step_jacobians_dt=ca.external("step_jacobians_dt", so_str),
        output_jacobians=ca.external("output_jacobians", so_str),
        f_ode=ca.external("f_ode", so_str),
    )


def benchmark_compiled_library(
    lib: Compiled14DofLibrary,
    model: Optional[DifferentiableVehicle14Dof] = None,
    n_iters: int = 2000,
) -> Dict[str, float]:
    """Benchmark step_dynamics and step_jacobians latency in microseconds."""
    if model is None:
        model = DifferentiableVehicle14Dof()
    x = model.default_state(vx=18.0, vy=0.4, wz=0.35)
    u = model.inputs_from_driver(delta_cmd=0.08, throttle=0.5, brake=0.0)
    theta = model.default_theta()
    w = model.default_disturbance()

    # Warmup
    for _ in range(50):
        lib.step_dynamics(x, u, theta, w)
        lib.step_jacobians(x, u, theta, w)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        lib.step_dynamics(x, u, theta, w)
    t_step_us = (time.perf_counter() - t0) * 1e6 / n_iters

    t1 = time.perf_counter()
    for _ in range(n_iters):
        lib.step_jacobians(x, u, theta, w)
    t_jac_us = (time.perf_counter() - t1) * 1e6 / n_iters

    return {
        "step_dynamics_us": float(t_step_us),
        "step_jacobians_us": float(t_jac_us),
        "n_iters": int(n_iters),
    }
