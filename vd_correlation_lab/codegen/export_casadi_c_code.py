#!/usr/bin/env python3
"""
Pure C99 Code Generation and Verification Harness for TeR27 4WD CasADi Model.

Uses casadi.CodeGenerator on ter27_casadi_model.py to export standalone,
dependency-free C99 code (ter27_casadi_eval.c and ter27_casadi_eval.h) for:
  1. ter27_eval_torque:
       Inputs:  (vx, vy, wz, delta, T_4[4], F_brk_hyd, ax_lt, ay_lt)
       Outputs: (Fx_tot, Fy_tot, Mz_tot, tire_util_4[4], P_dc_kw, J_T[4x4])
       where J_T = d[Fx_tot, Fy_tot, Mz_tot, P_dc_kw] / dT_4 (4x4 exact Jacobian).
  2. ter27_eval_kappa:
       Inputs:  (vx, vy, wz, delta, kappa_4[4], F_brk_hyd, ax_lt, ay_lt)
       Outputs: (Fx_tot, Fy_tot, Mz_tot, tire_util_4[4], P_dc_kw, T_motor_4[4], J_T_ift[4x4])
       where J_T_ift = (dy/dkappa) * diag(1 / (dT_motor_i/dkappa_i)) via Implicit Function Theorem.

Compiles a standalone C99 verification & latency benchmark harness with `gcc -O3 -lm`
(zero external libraries beyond libm), checks numerical accuracy < 1e-9 vs Python CasADi
across >= 50 realistic operating points, and measures mean and p99 latency in microseconds.

Outputs in vd_correlation_lab/results/casadi_suite/:
  - ter27_casadi_eval.c
  - ter27_casadi_eval.h
  - c_codegen_report.json
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import casadi as ca
import numpy as np

CODEGEN_DIR = Path(__file__).resolve().parent
LAB_DIR = CODEGEN_DIR.parent
REPO_ROOT = LAB_DIR.parent
OUT_DIR = LAB_DIR / "results" / "casadi_suite"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vd_correlation_lab.models.vehicle_14dof.ter27_casadi_model import build_casadi_vehicle, sabs, sclip, smax, smin, spos  # noqa: E402


def _compute_tire_util_4(veh: Any, out: Dict[str, Any]) -> ca.SX:
    """
    Smooth C^2 tire friction ellipse utilization per wheel:
      u_i = sqrt((Fx_w_i / F_lim_x_i)^2 + (Fy_w_i / F_lim_y_i)^2 + eps)
    where F_lim_x_i = mu_x_i * Fz_i and F_lim_y_i = mu_y_i * Fz_i (MF 6.2 peak grip).
    """
    t = veh.tire
    s = veh.smoothing
    p_infl = veh.params.p_infl_kpa

    fz = out["Fz_4"]
    fx_w = out["Fx_w_4"]
    fy_w = out["Fy_w_4"]
    delta_w = out["delta_w_4"]
    phi = out["phi"]
    gamma = veh.cambers_smooth(delta_w, phi)

    fz_pos = spos(fz, s["eps_fz_n"])
    dfz_raw = (fz_pos - t.fnomin) / t.fnomin
    if math.isfinite(t.dfz_max):
        dfz = smin(dfz_raw, t.dfz_max, s["eps_dfz"])
    else:
        dfz = dfz_raw
    dpi = (p_infl - t.pinom) / t.pinom
    gamma_sq = gamma * gamma

    camber_f_x = smax(1.0 - t.PDX3 * gamma_sq, 0.2, s["eps_camber_f"])
    mu_x = (
        (t.PDX1 + t.PDX2 * dfz)
        * (1.0 + t.PPX3 * dpi + t.PPX4 * dpi * dpi)
        * camber_f_x
        * veh.lmux
    )
    camber_f_y = smax(1.0 - t.PDY3 * gamma_sq, 0.2, s["eps_camber_f"])
    mu_y = (
        (t.PDY1 + t.PDY2 * dfz)
        * (1.0 + t.PPY3 * dpi + t.PPY4 * dpi * dpi)
        * camber_f_y
        * veh.lmuy
    )

    fx_lim = smax(mu_x * fz_pos, 50.0, 5.0)
    fy_lim = smax(mu_y * fz_pos, 50.0, 5.0)
    ux = fx_w / fx_lim
    uy = fy_w / fy_lim
    return ca.sqrt(ux * ux + uy * uy + 1e-12)


def build_export_functions(grip_profile: str = "FSG_REAL_TRACK") -> Tuple[ca.Function, ca.Function, Any]:
    """
    Build the two symbolic CasADi SX functions to export to C99:
      - ter27_eval_torque: directly parameterized by motor torques T_4[4]
      - ter27_eval_kappa:  parameterized by wheel slip ratios kappa_4[4] with IFT Jacobian dY/dT_4
    """
    veh = build_casadi_vehicle(grip_profile=grip_profile)

    s_vx = ca.SX.sym("vx")
    s_vy = ca.SX.sym("vy")
    s_wz = ca.SX.sym("wz")
    s_del = ca.SX.sym("delta")
    s_T4 = ca.SX.sym("T_4", 4)
    s_kap = ca.SX.sym("kappa_4", 4)
    s_fbrk = ca.SX.sym("F_brk_hyd")
    s_ax = ca.SX.sym("ax_lt")
    s_ay = ca.SX.sym("ay_lt")

    # 1. Torque-parameterized evaluation (with exact AD Jacobian d(Fx, Fy, Mz, P_dc)/dT_4)
    out_t = veh.eval_forces_and_power(s_vx, s_vy, s_wz, s_del, s_T4, s_fbrk, s_ax, s_ay)
    util_t = _compute_tire_util_4(veh, out_t)
    y_t = ca.vertcat(out_t["Fx_tot"], out_t["Fy_tot"], out_t["Mz_tot"], out_t["P_dc_kw"])
    J_T = ca.jacobian(y_t, s_T4)  # 4 x 4 exact Jacobian

    fn_eval_torque = ca.Function(
        "ter27_eval_torque",
        [s_vx, s_vy, s_wz, s_del, s_T4, s_fbrk, s_ax, s_ay],
        [
            out_t["Fx_tot"],
            out_t["Fy_tot"],
            out_t["Mz_tot"],
            util_t,
            out_t["P_dc_kw"],
            J_T,
        ],
        ["vx", "vy", "wz", "delta", "T_4", "F_brk_hyd", "ax_lt", "ay_lt"],
        ["Fx_tot", "Fy_tot", "Mz_tot", "tire_util_4", "P_dc_kw", "J_T"],
    )

    # 2. Slip-parameterized evaluation (ultra-fast for 200 Hz MCU + IFT Jacobian d(Fx, Fy, Mz, P_dc)/dT_4)
    out_k = veh.eval_forces_from_kappa(s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay)
    util_k = _compute_tire_util_4(veh, out_k)
    y_k = ca.vertcat(out_k["Fx_tot"], out_k["Fy_tot"], out_k["Mz_tot"], out_k["P_dc_kw"])
    J_y_kap = ca.jacobian(y_k, s_kap)  # 4 x 4
    # T_motor_i depends only on kappa_i when (ax_lt, ay_lt) are held fixed -> diagonal dT/dkappa
    dT_dkap_diag = ca.diag(ca.jacobian(out_k["T_motor_4"], s_kap))  # 4 x 1
    dT_dkap_safe = smax(sabs(dT_dkap_diag, 1e-3), 1.0, 1e-2)
    # J_T_ift[:, j] = J_y_kap[:, j] / dT_dkap_diag[j]
    J_T_ift = J_y_kap @ ca.diag(1.0 / dT_dkap_safe)

    fn_eval_kappa = ca.Function(
        "ter27_eval_kappa",
        [s_vx, s_vy, s_wz, s_del, s_kap, s_fbrk, s_ax, s_ay],
        [
            out_k["Fx_tot"],
            out_k["Fy_tot"],
            out_k["Mz_tot"],
            util_k,
            out_k["P_dc_kw"],
            out_k["T_motor_4"],
            J_T_ift,
        ],
        ["vx", "vy", "wz", "delta", "kappa_4", "F_brk_hyd", "ax_lt", "ay_lt"],
        ["Fx_tot", "Fy_tot", "Mz_tot", "tire_util_4", "P_dc_kw", "T_motor_4", "J_T_ift"],
    )

    return fn_eval_torque, fn_eval_kappa, veh


def generate_c99_code(
    fn_eval_torque: ca.Function,
    fn_eval_kappa: ca.Function,
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Generate standalone C99 source and header using casadi.CodeGenerator."""
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "with_header": True,
        "with_mem": False,
        "main": False,
        "cpp": False,
        "casadi_real": "double",
        "casadi_int": "long long int",
    }
    gen = ca.CodeGenerator("ter27_casadi_eval.c", opts)
    gen.add(fn_eval_torque)
    gen.add(fn_eval_kappa)

    cwd = os.getcwd()
    try:
        os.chdir(str(out_dir))
        gen.generate()
    finally:
        os.chdir(cwd)

    c_path = out_dir / "ter27_casadi_eval.c"
    h_path = out_dir / "ter27_casadi_eval.h"
    if not c_path.exists() or not h_path.exists():
        raise RuntimeError(f"CodeGenerator failed to create {c_path} / {h_path}")
    return c_path, h_path


def generate_operating_points(n_points: int = 64, seed: int = 2026) -> List[Dict[str, Any]]:
    """
    Generate diverse, physically realistic TeR27 4WD operating points covering:
      - straight-line acceleration (low/high speed)
      - high-speed cornering + aero load
      - tight hairpin with asymmetric TV & inner regen
      - heavy trail-braking with hydraulic + regen blending
    """
    rng = np.random.default_rng(seed)
    pts: List[Dict[str, Any]] = []
    for i in range(n_points):
        regime = i % 4
        if regime == 0:
            # Straight / fast sweep acceleration
            vx = float(rng.uniform(8.0, 30.0))
            vy = float(rng.uniform(-0.3, 0.3))
            wz = float(rng.uniform(-0.4, 0.4))
            delta = float(rng.uniform(-0.08, 0.08))
            T_4 = rng.uniform(40.0, 220.0, size=4)
            F_brk = 0.0
            ax = float(rng.uniform(2.0, 11.0))
            ay = float(rng.uniform(-5.0, 5.0))
        elif regime == 1:
            # Steady / high-speed cornering with TV
            vx = float(rng.uniform(14.0, 26.0))
            vy = float(rng.uniform(-1.0, 1.0))
            wz = float(rng.uniform(-1.5, 1.5))
            delta = float(rng.uniform(-0.22, 0.22))
            T_4 = np.array([
                rng.uniform(-40.0, 120.0),
                rng.uniform(-40.0, 120.0),
                rng.uniform(-40.0, 160.0),
                rng.uniform(-40.0, 160.0),
            ])
            F_brk = 0.0
            ax = float(rng.uniform(-3.0, 6.0))
            ay = float(rng.uniform(-14.0, 14.0))
        elif regime == 2:
            # Tight hairpin apex / exit with strong L/R torque vectoring
            vx = float(rng.uniform(7.0, 13.0))
            vy = float(rng.uniform(-0.8, 0.8))
            wz = float(rng.uniform(-2.2, 2.2))
            delta = float(rng.uniform(-0.35, 0.35))
            sign = 1.0 if delta >= 0.0 else -1.0
            T_4 = np.array([
                rng.uniform(-80.0, 60.0) if sign > 0 else rng.uniform(60.0, 200.0),
                rng.uniform(60.0, 200.0) if sign > 0 else rng.uniform(-80.0, 60.0),
                rng.uniform(-90.0, 80.0) if sign > 0 else rng.uniform(80.0, 220.0),
                rng.uniform(80.0, 220.0) if sign > 0 else rng.uniform(-90.0, 80.0),
            ])
            F_brk = 0.0
            ax = float(rng.uniform(-4.0, 8.0))
            ay = float(rng.uniform(-13.0, 13.0))
        else:
            # Heavy braking / trail-braking (FSAE BSPD compliant: T_4 <= 0 when F_brk > 0)
            vx = float(rng.uniform(10.0, 29.0))
            vy = float(rng.uniform(-0.6, 0.6))
            wz = float(rng.uniform(-1.2, 1.2))
            delta = float(rng.uniform(-0.20, 0.20))
            T_4 = rng.uniform(-110.0, -5.0, size=4)
            F_brk = float(rng.uniform(500.0, 4500.0))
            ax = float(rng.uniform(-16.0, -4.0))
            ay = float(rng.uniform(-10.0, 10.0))

        kappa_4 = rng.uniform(-0.08, 0.08, size=4)
        pts.append({
            "vx": vx,
            "vy": vy,
            "wz": wz,
            "delta": delta,
            "T_4": T_4.tolist(),
            "kappa_4": kappa_4.tolist(),
            "F_brk_hyd": F_brk,
            "ax_lt": ax,
            "ay_lt": ay,
        })
    return pts


HARNESS_C_SRC = r"""
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ter27_casadi_eval.h"

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <inputs.txt> <outputs.txt> <bench_iters>\n", argv[0]);
        return 1;
    }
    const char *in_path = argv[1];
    const char *out_path = argv[2];
    int bench_iters = atoi(argv[3]);
    if (bench_iters < 100) bench_iters = 100;

    FILE *fin = fopen(in_path, "r");
    if (!fin) { perror("fopen inputs"); return 1; }
    int n_pts = 0;
    if (fscanf(fin, "%d", &n_pts) != 1 || n_pts <= 0) {
        fprintf(stderr, "Invalid point count\n");
        fclose(fin);
        return 1;
    }

    double *in_buf = (double *)malloc((size_t)n_pts * 15 * sizeof(double));
    if (!in_buf) { fclose(fin); return 1; }
    for (int i = 0; i < n_pts; ++i) {
        for (int k = 0; k < 15; ++k) {
            if (fscanf(fin, "%lf", &in_buf[i * 15 + k]) != 1) {
                fprintf(stderr, "Failed reading point %d col %d\n", i, k);
                free(in_buf);
                fclose(fin);
                return 1;
            }
        }
    }
    fclose(fin);

    // Query workspace sizes
    casadi_int sz_arg_t = 0, sz_res_t = 0, sz_iw_t = 0, sz_w_t = 0;
    ter27_eval_torque_work(&sz_arg_t, &sz_res_t, &sz_iw_t, &sz_w_t);
    casadi_int sz_arg_k = 0, sz_res_k = 0, sz_iw_k = 0, sz_w_k = 0;
    ter27_eval_kappa_work(&sz_arg_k, &sz_res_k, &sz_iw_k, &sz_w_k);

    casadi_int sz_arg = sz_arg_t > sz_arg_k ? sz_arg_t : sz_arg_k;
    casadi_int sz_res = sz_res_t > sz_res_k ? sz_res_t : sz_res_k;
    casadi_int sz_iw  = sz_iw_t  > sz_iw_k  ? sz_iw_t  : sz_iw_k;
    casadi_int sz_w   = sz_w_t   > sz_w_k   ? sz_w_t   : sz_w_k;

    const double **arg = (const double **)calloc((size_t)(sz_arg + 1), sizeof(double *));
    double **res = (double **)calloc((size_t)(sz_res + 1), sizeof(double *));
    casadi_int *iw = (casadi_int *)calloc((size_t)(sz_iw + 1), sizeof(casadi_int));
    double *w = (double *)calloc((size_t)(sz_w + 1), sizeof(double));

    // 1. Accuracy evaluation across all n_pts
    FILE *fout = fopen(out_path, "w");
    if (!fout) { perror("fopen outputs"); return 1; }

    for (int i = 0; i < n_pts; ++i) {
        const double *row = &in_buf[i * 15];
        double vx = row[0], vy = row[1], wz = row[2], delta = row[3];
        double T_4[4] = { row[4], row[5], row[6], row[7] };
        double kap_4[4] = { row[8], row[9], row[10], row[11] };
        double fbrk = row[12], ax = row[13], ay = row[14];

        // Evaluate ter27_eval_torque
        double fx_t = 0.0, fy_t = 0.0, mz_t = 0.0, pdc_t = 0.0;
        double util_t[4] = {0}, JT[16] = {0};
        arg[0] = &vx; arg[1] = &vy; arg[2] = &wz; arg[3] = &delta;
        arg[4] = T_4; arg[5] = &fbrk; arg[6] = &ax; arg[7] = &ay;
        res[0] = &fx_t; res[1] = &fy_t; res[2] = &mz_t;
        res[3] = util_t; res[4] = &pdc_t; res[5] = JT;
        ter27_eval_torque(arg, res, iw, w, 0);

        // Evaluate ter27_eval_kappa
        double fx_k = 0.0, fy_k = 0.0, mz_k = 0.0, pdc_k = 0.0;
        double util_k[4] = {0}, tmot_k[4] = {0}, JT_k[16] = {0};
        arg[0] = &vx; arg[1] = &vy; arg[2] = &wz; arg[3] = &delta;
        arg[4] = kap_4; arg[5] = &fbrk; arg[6] = &ax; arg[7] = &ay;
        res[0] = &fx_k; res[1] = &fy_k; res[2] = &mz_k;
        res[3] = util_k; res[4] = &pdc_k; res[5] = tmot_k; res[6] = JT_k;
        ter27_eval_kappa(arg, res, iw, w, 0);

        // Write torque outputs (24 doubles) + kappa outputs (28 doubles)
        fprintf(fout, "%.17g %.17g %.17g %.17g", fx_t, fy_t, mz_t, pdc_t);
        for (int j = 0; j < 4; ++j) fprintf(fout, " %.17g", util_t[j]);
        for (int j = 0; j < 16; ++j) fprintf(fout, " %.17g", JT[j]);

        fprintf(fout, " %.17g %.17g %.17g %.17g", fx_k, fy_k, mz_k, pdc_k);
        for (int j = 0; j < 4; ++j) fprintf(fout, " %.17g", util_k[j]);
        for (int j = 0; j < 4; ++j) fprintf(fout, " %.17g", tmot_k[j]);
        for (int j = 0; j < 16; ++j) fprintf(fout, " %.17g", JT_k[j]);
        fprintf(fout, "\n");
    }
    fclose(fout);

    // 2. Latency benchmark (warmup + timed samples)
    double *lat_t = (double *)malloc((size_t)bench_iters * sizeof(double));
    double *lat_k = (double *)malloc((size_t)bench_iters * sizeof(double));
    volatile double sink = 0.0;

    // Warmup
    for (int it = 0; it < 200; ++it) {
        const double *row = &in_buf[(it % n_pts) * 15];
        double vx = row[0], vy = row[1], wz = row[2], delta = row[3];
        double T_4[4] = { row[4], row[5], row[6], row[7] };
        double kap_4[4] = { row[8], row[9], row[10], row[11] };
        double fbrk = row[12], ax = row[13], ay = row[14];
        double fx_t, fy_t, mz_t, pdc_t, util_t[4], JT[16];
        arg[0] = &vx; arg[1] = &vy; arg[2] = &wz; arg[3] = &delta;
        arg[4] = T_4; arg[5] = &fbrk; arg[6] = &ax; arg[7] = &ay;
        res[0] = &fx_t; res[1] = &fy_t; res[2] = &mz_t;
        res[3] = util_t; res[4] = &pdc_t; res[5] = JT;
        ter27_eval_torque(arg, res, iw, w, 0);
        sink += fx_t;

        double fx_k, fy_k, mz_k, pdc_k, util_k[4], tmot_k[4], JT_k[16];
        arg[4] = kap_4;
        res[0] = &fx_k; res[1] = &fy_k; res[2] = &mz_k;
        res[3] = util_k; res[4] = &pdc_k; res[5] = tmot_k; res[6] = JT_k;
        ter27_eval_kappa(arg, res, iw, w, 0);
        sink += fx_k;
    }

    // Measure torque eval
    for (int it = 0; it < bench_iters; ++it) {
        const double *row = &in_buf[(it % n_pts) * 15];
        double vx = row[0], vy = row[1], wz = row[2], delta = row[3];
        double T_4[4] = { row[4], row[5], row[6], row[7] };
        double fbrk = row[12], ax = row[13], ay = row[14];
        double fx_t, fy_t, mz_t, pdc_t, util_t[4], JT[16];
        arg[0] = &vx; arg[1] = &vy; arg[2] = &wz; arg[3] = &delta;
        arg[4] = T_4; arg[5] = &fbrk; arg[6] = &ax; arg[7] = &ay;
        res[0] = &fx_t; res[1] = &fy_t; res[2] = &mz_t;
        res[3] = util_t; res[4] = &pdc_t; res[5] = JT;

        double t0 = now_us();
        ter27_eval_torque(arg, res, iw, w, 0);
        double t1 = now_us();
        lat_t[it] = t1 - t0;
        sink += fx_t + JT[0];
    }

    // Measure kappa eval
    for (int it = 0; it < bench_iters; ++it) {
        const double *row = &in_buf[(it % n_pts) * 15];
        double vx = row[0], vy = row[1], wz = row[2], delta = row[3];
        double kap_4[4] = { row[8], row[9], row[10], row[11] };
        double fbrk = row[12], ax = row[13], ay = row[14];
        double fx_k, fy_k, mz_k, pdc_k, util_k[4], tmot_k[4], JT_k[16];
        arg[0] = &vx; arg[1] = &vy; arg[2] = &wz; arg[3] = &delta;
        arg[4] = kap_4; arg[5] = &fbrk; arg[6] = &ax; arg[7] = &ay;
        res[0] = &fx_k; res[1] = &fy_k; res[2] = &mz_k;
        res[3] = util_k; res[4] = &pdc_k; res[5] = tmot_k; res[6] = JT_k;

        double t0 = now_us();
        ter27_eval_kappa(arg, res, iw, w, 0);
        double t1 = now_us();
        lat_k[it] = t1 - t0;
        sink += fx_k + JT_k[0];
    }

    double sum_t = 0.0, sum_k = 0.0;
    for (int it = 0; it < bench_iters; ++it) {
        sum_t += lat_t[it];
        sum_k += lat_k[it];
    }
    qsort(lat_t, (size_t)bench_iters, sizeof(double), cmp_double);
    qsort(lat_k, (size_t)bench_iters, sizeof(double), cmp_double);

    int idx_p50 = (int)(0.50 * (bench_iters - 1));
    int idx_p95 = (int)(0.95 * (bench_iters - 1));
    int idx_p99 = (int)(0.99 * (bench_iters - 1));

    printf("BENCH_JSON:{\"bench_iters\":%d,\"sink\":%.6f,"
           "\"torque_mean_us\":%.4f,\"torque_p50_us\":%.4f,\"torque_p95_us\":%.4f,\"torque_p99_us\":%.4f,"
           "\"kappa_mean_us\":%.4f,\"kappa_p50_us\":%.4f,\"kappa_p95_us\":%.4f,\"kappa_p99_us\":%.4f,"
           "\"torque_work_doubles\":%lld,\"kappa_work_doubles\":%lld}\n",
           bench_iters, sink,
           sum_t / bench_iters, lat_t[idx_p50], lat_t[idx_p95], lat_t[idx_p99],
           sum_k / bench_iters, lat_k[idx_p50], lat_k[idx_p95], lat_k[idx_p99],
           (long long)sz_w_t, (long long)sz_w_k);

    free(lat_t); free(lat_k); free(in_buf);
    free(arg); free(res); free(iw); free(w);
    return 0;
}
"""


def run_c_codegen_and_verify(
    grip_profile: str = "FSG_REAL_TRACK",
    n_points: int = 64,
    bench_iters: int = 5000,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    print(f"[C99 CODEGEN] Building CasADi SX functions ({grip_profile})...", flush=True)
    fn_eval_torque, fn_eval_kappa, _ = build_export_functions(grip_profile=grip_profile)

    print(f"[C99 CODEGEN] Exporting standalone C99 to {OUT_DIR}...", flush=True)
    c_path, h_path = generate_c99_code(fn_eval_torque, fn_eval_kappa, OUT_DIR)
    c_size_kb = c_path.stat().st_size / 1024.0
    h_size_kb = h_path.stat().st_size / 1024.0
    print(
        f"[C99 CODEGEN] Generated {c_path.name} ({c_size_kb:.1f} KB) and {h_path.name} ({h_size_kb:.1f} KB)",
        flush=True,
    )

    pts = generate_operating_points(n_points=n_points)

    # Compile and execute standalone C99 benchmark/verification harness
    with tempfile.TemporaryDirectory(prefix="ter27_c99_") as tmp:
        tmp_dir = Path(tmp)
        harness_c = tmp_dir / "bench_harness.c"
        harness_bin = tmp_dir / "bench_harness"
        in_txt = tmp_dir / "inputs.txt"
        out_txt = tmp_dir / "outputs.txt"

        harness_c.write_text(HARNESS_C_SRC, encoding="utf-8")

        # Write inputs
        lines = [str(len(pts))]
        for pt in pts:
            vals = (
                [pt["vx"], pt["vy"], pt["wz"], pt["delta"]]
                + pt["T_4"]
                + pt["kappa_4"]
                + [pt["F_brk_hyd"], pt["ax_lt"], pt["ay_lt"]]
            )
            lines.append(" ".join(f"{v:.17g}" for v in vals))
        in_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Compile with gcc -std=c99 -O3 (only -lm)
        compile_cmd = [
            "/usr/bin/gcc",
            "-std=c99",
            "-O3",
            "-Wall",
            "-Wno-unused-variable",
            "-Wno-unused-function",
            f"-I{OUT_DIR}",
            str(c_path),
            str(harness_c),
            "-o",
            str(harness_bin),
            "-lm",
        ]
        print(f"[C99 CODEGEN] Compiling with: {' '.join(compile_cmd)}", flush=True)
        t_cc0 = time.perf_counter()
        cc_proc = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
        t_cc = time.perf_counter() - t_cc0
        if cc_proc.returncode != 0:
            raise RuntimeError(
                f"gcc compilation failed (rc={cc_proc.returncode}):\n"
                f"STDOUT:\n{cc_proc.stdout}\nSTDERR:\n{cc_proc.stderr}"
            )
        print(f"[C99 CODEGEN] Compiled in {t_cc:.2f}s (warnings={len(cc_proc.stderr.splitlines())})", flush=True)

        # Run harness
        run_proc = subprocess.run(
            [str(harness_bin), str(in_txt), str(out_txt), str(bench_iters)],
            capture_output=True,
            text=True,
            check=False,
        )
        if run_proc.returncode != 0:
            raise RuntimeError(
                f"C harness execution failed (rc={run_proc.returncode}):\n"
                f"STDOUT:\n{run_proc.stdout}\nSTDERR:\n{run_proc.stderr}"
            )

        bench_json_line = ""
        for line in run_proc.stdout.splitlines():
            if line.startswith("BENCH_JSON:"):
                bench_json_line = line[len("BENCH_JSON:") :]
                break
        if not bench_json_line:
            raise RuntimeError(f"Missing BENCH_JSON in harness output:\n{run_proc.stdout}")
        bench = json.loads(bench_json_line)

        # Parse C outputs and compare against Python CasADi
        c_rows = [
            [float(x) for x in ln.strip().split()]
            for ln in out_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(c_rows) == n_points

    max_abs_err_torque = 0.0
    max_rel_err_torque = 0.0
    max_abs_err_kappa = 0.0
    max_rel_err_kappa = 0.0
    per_output_err_torque: Dict[str, float] = {
        "Fx_tot": 0.0,
        "Fy_tot": 0.0,
        "Mz_tot": 0.0,
        "P_dc_kw": 0.0,
        "tire_util_4": 0.0,
        "J_T": 0.0,
    }
    per_output_err_kappa: Dict[str, float] = {
        "Fx_tot": 0.0,
        "Fy_tot": 0.0,
        "Mz_tot": 0.0,
        "P_dc_kw": 0.0,
        "tire_util_4": 0.0,
        "T_motor_4": 0.0,
        "J_T_ift": 0.0,
    }

    for i, pt in enumerate(pts):
        row = c_rows[i]
        # Torque outputs from C (column-major for J_T in CasADi!)
        c_fx_t, c_fy_t, c_mz_t, c_pdc_t = row[0], row[1], row[2], row[3]
        c_util_t = np.array(row[4:8], dtype=np.float64)
        c_JT = np.array(row[8:24], dtype=np.float64).reshape((4, 4), order="F")

        py_t = fn_eval_torque(
            pt["vx"], pt["vy"], pt["wz"], pt["delta"],
            pt["T_4"], pt["F_brk_hyd"], pt["ax_lt"], pt["ay_lt"],
        )
        p_fx_t = float(py_t[0])
        p_fy_t = float(py_t[1])
        p_mz_t = float(py_t[2])
        p_util_t = np.array(py_t[3].full()).ravel()
        p_pdc_t = float(py_t[4])
        p_JT = np.array(py_t[5].full())

        pairs_t = [
            ("Fx_tot", c_fx_t, p_fx_t),
            ("Fy_tot", c_fy_t, p_fy_t),
            ("Mz_tot", c_mz_t, p_mz_t),
            ("P_dc_kw", c_pdc_t, p_pdc_t),
            ("tire_util_4", c_util_t, p_util_t),
            ("J_T", c_JT, p_JT),
        ]
        for name, cv, pv in pairs_t:
            diff = np.abs(np.asarray(cv) - np.asarray(pv))
            ae = float(np.max(diff))
            re = float(np.max(diff / np.maximum(np.abs(np.asarray(pv)), 1.0)))
            per_output_err_torque[name] = max(per_output_err_torque[name], ae)
            max_abs_err_torque = max(max_abs_err_torque, ae)
            max_rel_err_torque = max(max_rel_err_torque, re)

        # Kappa outputs from C
        off = 24
        c_fx_k, c_fy_k, c_mz_k, c_pdc_k = row[off], row[off + 1], row[off + 2], row[off + 3]
        c_util_k = np.array(row[off + 4 : off + 8], dtype=np.float64)
        c_tmot_k = np.array(row[off + 8 : off + 12], dtype=np.float64)
        c_JTk = np.array(row[off + 12 : off + 28], dtype=np.float64).reshape((4, 4), order="F")

        py_k = fn_eval_kappa(
            pt["vx"], pt["vy"], pt["wz"], pt["delta"],
            pt["kappa_4"], pt["F_brk_hyd"], pt["ax_lt"], pt["ay_lt"],
        )
        p_fx_k = float(py_k[0])
        p_fy_k = float(py_k[1])
        p_mz_k = float(py_k[2])
        p_util_k = np.array(py_k[3].full()).ravel()
        p_pdc_k = float(py_k[4])
        p_tmot_k = np.array(py_k[5].full()).ravel()
        p_JTk = np.array(py_k[6].full())

        pairs_k = [
            ("Fx_tot", c_fx_k, p_fx_k),
            ("Fy_tot", c_fy_k, p_fy_k),
            ("Mz_tot", c_mz_k, p_mz_k),
            ("P_dc_kw", c_pdc_k, p_pdc_k),
            ("tire_util_4", c_util_k, p_util_k),
            ("T_motor_4", c_tmot_k, p_tmot_k),
            ("J_T_ift", c_JTk, p_JTk),
        ]
        for name, cv, pv in pairs_k:
            diff = np.abs(np.asarray(cv) - np.asarray(pv))
            ae = float(np.max(diff))
            re = float(np.max(diff / np.maximum(np.abs(np.asarray(pv)), 1.0)))
            per_output_err_kappa[name] = max(per_output_err_kappa[name], ae)
            max_abs_err_kappa = max(max_abs_err_kappa, ae)
            max_rel_err_kappa = max(max_rel_err_kappa, re)

    # Check GCC version
    gcc_ver = subprocess.run(["/usr/bin/gcc", "--version"], capture_output=True, text=True, check=False).stdout.splitlines()[0]

    acc_pass = (max_abs_err_torque < 1e-9) and (max_abs_err_kappa < 1e-9)
    lat_pass = (bench["torque_p99_us"] < 50.0) and (bench["kappa_p99_us"] < 50.0)

    report: Dict[str, Any] = {
        "grip_profile": grip_profile,
        "compiler": gcc_ver,
        "compile_flags": "-std=c99 -O3 -lm",
        "compile_time_s": round(t_cc, 3),
        "generated_files": {
            "c_source": str(c_path.relative_to(LAB_DIR.parent)),
            "h_header": str(h_path.relative_to(LAB_DIR.parent)),
            "c_size_kb": round(c_size_kb, 2),
            "h_size_kb": round(h_size_kb, 2),
        },
        "verification": {
            "n_operating_points": n_points,
            "torque_eval_max_abs_err": max_abs_err_torque,
            "torque_eval_max_rel_err": max_rel_err_torque,
            "torque_eval_per_output_max_abs_err": per_output_err_torque,
            "kappa_eval_max_abs_err": max_abs_err_kappa,
            "kappa_eval_max_rel_err": max_rel_err_kappa,
            "kappa_eval_per_output_max_abs_err": per_output_err_kappa,
            "accuracy_target": 1e-9,
            "accuracy_pass": bool(acc_pass),
        },
        "latency_us": {
            "bench_iters": int(bench["bench_iters"]),
            "ter27_eval_torque": {
                "mean_us": round(float(bench["torque_mean_us"]), 4),
                "p50_us": round(float(bench["torque_p50_us"]), 4),
                "p95_us": round(float(bench["torque_p95_us"]), 4),
                "p99_us": round(float(bench["torque_p99_us"]), 4),
                "work_buffer_doubles": int(bench["torque_work_doubles"]),
                "work_buffer_bytes": int(bench["torque_work_doubles"]) * 8,
            },
            "ter27_eval_kappa": {
                "mean_us": round(float(bench["kappa_mean_us"]), 4),
                "p50_us": round(float(bench["kappa_p50_us"]), 4),
                "p95_us": round(float(bench["kappa_p95_us"]), 4),
                "p99_us": round(float(bench["kappa_p99_us"]), 4),
                "work_buffer_doubles": int(bench["kappa_work_doubles"]),
                "work_buffer_bytes": int(bench["kappa_work_doubles"]) * 8,
            },
            "p99_target_us": 50.0,
            "latency_pass": bool(lat_pass),
            "stm32f405_200hz_budget_us": 5000.0,
        },
        "overall_pass": bool(acc_pass and lat_pass),
        "total_runtime_s": round(time.perf_counter() - t0, 3),
    }

    rep_path = OUT_DIR / "c_codegen_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[C99 CODEGEN] DONE: overall_pass={report['overall_pass']} | "
        f"torque_err={max_abs_err_torque:.3e}, p99={bench['torque_p99_us']:.2f}us (mean={bench['torque_mean_us']:.2f}us) | "
        f"kappa_err={max_abs_err_kappa:.3e}, p99={bench['kappa_p99_us']:.2f}us (mean={bench['kappa_mean_us']:.2f}us)",
        flush=True,
    )
    print(f"[C99 CODEGEN] Saved report: {rep_path}", flush=True)
    return report


if __name__ == "__main__":
    run_c_codegen_and_verify()
