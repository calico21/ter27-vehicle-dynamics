#!/usr/bin/env python3
"""
Optimality Gap & Surgical Ablation evaluator for EcuTorqueVectoring4Wd (TeR27 4WD).
Runs the real 14-DOF vehicle model in headless Godot on:
  - Stationary corners R = 5.0, 9.125, 15.0, 25.0 m (FSG_REAL_TRACK & TTC_LAB_PEAK)
  - Transient 180 deg hairpin (R = 5.5 m, entry 20 m/s, exit 40 m straight)
  - 55 km/h slalom (max|beta|)
  - 5-map HV power audit ([-25.0, +80.0] kW) and solver step_time_p99.

Compares against Oracle reports (stationary_oracle_report.json, hairpin_ocp_report.json)
when available.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ORACLE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "vd_correlation_lab" / "results" / "tv_oracle"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from vd_correlation_lab.config.paths import GODOT_BIN as _GODOT_BIN  # noqa: E402
GODOT_BIN = str(_GODOT_BIN)
PROBE_GD = ORACLE_DIR / "godot_gap_probe.gd"


def run_probe(cfg_path: str | None = None, ablation_mask: int = 0, timeout_s: int = 180) -> dict:
    with tempfile.TemporaryDirectory(prefix="tv_gap_probe_") as tmp:
        out_json = Path(tmp) / "probe_out.json"
        cmd = [
            GODOT_BIN,
            "--headless",
            "--path",
            str(REPO_ROOT / "src" / "mirena_sim"),
            "-s",
            str(PROBE_GD),
            "--",
            str(out_json),
            str(cfg_path) if cfg_path else "default",
            str(int(ablation_mask)),
        ]
        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.time() - t0
        if res.returncode != 0 or not out_json.exists():
            raise RuntimeError(
                f"Godot probe failed (rc={res.returncode}):\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )
        data = json.loads(out_json.read_text(encoding="utf-8"))
        data["wall_time_s"] = round(elapsed, 2)
        return data


def _extract_oracle_stationary(oracle_path: Path) -> dict:
    if not oracle_path.exists():
        return {}
    raw = json.loads(oracle_path.read_text(encoding="utf-8"))
    out: dict = {}
    profiles = raw.get("profiles", raw)
    for prof in ("FSG_REAL_TRACK", "TTC_LAB_PEAK"):
        entries = profiles.get(prof)
        out[prof] = {}
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict) and "radius_m" in item:
                    key = f"R_{float(item['radius_m']):.3f}"
                    out[prof][key] = item
        elif isinstance(entries, dict):
            out[prof] = entries
    return out


def compute_gap_summary(probe: dict, stat_oracle_path: Path, hp_oracle_path: Path) -> dict:
    gap: dict = {"stationary_gap": {}, "hairpin_gap": {}}
    stat_or = _extract_oracle_stationary(stat_oracle_path)
    for prof in ("FSG_REAL_TRACK", "TTC_LAB_PEAK"):
        p_prof = probe.get("stationary", {}).get(prof, {})
        o_prof = stat_or.get(prof, {})
        gap["stationary_gap"][prof] = {}
        pcts = []
        for r_val in (5.0, 9.125, 15.0, 25.0):
            key = f"R_{r_val:.3f}"
            p_ay = float(p_prof.get(key, {}).get("ay_ms2", 0.0))
            p_vx = float(p_prof.get(key, {}).get("vx_ms", 0.0))
            o_entry = o_prof.get(key)
            o_ay = None
            o_vx = None
            if isinstance(o_entry, dict):
                o_ay = float(o_entry.get("a_y_mps2", o_entry.get("ay_ms2", 0.0)))
                o_vx = float(o_entry.get("v_x_mps", o_entry.get("vx_ms", 0.0)))
            pct = (100.0 * p_ay / o_ay) if (o_ay and o_ay > 1e-3) else None
            if pct is not None:
                pcts.append(pct)
            gap["stationary_gap"][prof][key] = {
                "ecu_vx_ms": round(p_vx, 3),
                "oracle_vx_ms": round(o_vx, 3) if o_vx is not None else None,
                "ecu_ay_ms2": round(p_ay, 3),
                "ecu_ay_g": round(p_ay / 9.81, 3),
                "oracle_ay_ms2": round(o_ay, 3) if o_ay is not None else None,
                "oracle_ay_g": round(o_ay / 9.81, 3) if o_ay is not None else None,
                "pct_of_oracle": round(pct, 2) if pct is not None else None,
            }
        if pcts:
            gap["stationary_gap"][prof]["mean_pct_of_oracle"] = round(sum(pcts) / len(pcts), 2)

    if hp_oracle_path.exists():
        hp_or = json.loads(hp_oracle_path.read_text(encoding="utf-8"))
        for prof in ("FSG_REAL_TRACK", "TTC_LAB_PEAK"):
            p_t = float(probe.get("hairpin_180", {}).get(prof, {}).get("t_total_s", -1.0))
            o_t = None
            if prof in hp_or and isinstance(hp_or[prof], dict):
                perf = hp_or[prof].get("performance", hp_or[prof])
                if "total_time_s" in perf:
                    o_t = float(perf["total_time_s"])
            gap["hairpin_gap"][prof] = {
                "ecu_t_s": round(p_t, 4),
                "oracle_t_s": round(o_t, 4) if o_t is not None else None,
                "delta_s": round(p_t - o_t, 4) if (o_t is not None and p_t > 0) else None,
            }
    return gap


def print_summary(label: str, probe: dict, gap: dict) -> None:
    print(f"\n=== OPTIMALITY GAP PROBE: {label} (wall={probe.get('wall_time_s')}s) ===", flush=True)
    for prof in ("FSG_REAL_TRACK", "TTC_LAB_PEAK"):
        print(f"  [{prof}] Stationary Corners:", flush=True)
        for r_val in (5.0, 9.125, 15.0, 25.0):
            key = f"R_{r_val:.3f}"
            s = probe["stationary"][prof].get(key, {})
            g = gap["stationary_gap"][prof].get(key, {})
            pct_str = f" | {g['pct_of_oracle']:.2f}% of Oracle ({g['oracle_ay_ms2']:.2f})" if g.get("pct_of_oracle") is not None else ""
            print(
                f"    R={r_val:6.3f}m: vx={s.get('vx_ms', 0.0):6.3f} m/s, ay={s.get('ay_ms2', 0.0):6.3f} m/s2 ({s.get('ay_g', 0.0):.3f}g), "
                f"delta={s.get('delta_deg', 0.0):5.2f}deg, beta={s.get('beta_deg', 0.0):5.2f}deg, "
                f"af={s.get('alpha_f_deg', 0.0):5.2f}, ar={s.get('alpha_r_deg', 0.0):5.2f}, "
                f"Mz={s.get('mz_tv_nm', 0.0):6.1f}Nm, P={s.get('p_dc_kw', 0.0):5.1f}kW{pct_str}",
                flush=True,
            )
        sk = probe["stationary"][prof].get("R_9.125_SKIDPAD", {})
        print(
            f"    R= 9.125m (SKIDPAD mode): vx={sk.get('vx_ms', 0.0):6.3f} m/s, ay={sk.get('ay_ms2', 0.0):6.3f} m/s2 ({sk.get('ay_g', 0.0):.3f}g), "
            f"beta={sk.get('beta_deg', 0.0):5.2f}deg, Mz={sk.get('mz_tv_nm', 0.0):6.1f}Nm",
            flush=True,
        )
    print("  180 deg Hairpin (R=5.5m, 22m entry @ 20 m/s, 40m exit):", flush=True)
    for prof in ("FSG_REAL_TRACK", "TTC_LAB_PEAK"):
        hp = probe["hairpin_180"][prof]
        hg = gap.get("hairpin_gap", {}).get(prof, {})
        or_str = f" | Oracle={hg['oracle_t_s']:.3f}s (gap=+{hg['delta_s']:.3f}s)" if hg.get("oracle_t_s") is not None else ""
        print(
            f"    [{prof}] t_total={hp['t_total_s']:.3f}s (turn={hp['t_turn_180_s']:.3f}s, exit40={hp['t_exit_40m_s']:.3f}s){or_str}, "
            f"v_apex={hp['v_min_apex_ms']:.2f}m/s, max_cte={hp['max_cte_m']:.2f}m, max|beta|={hp['max_beta_deg']:.2f}deg, "
            f"T_in/out_apex={hp['apex_t_in_avg_nm']:.1f}/{hp['apex_t_out_avg_nm']:.1f}Nm, P=[{hp['p_dc_min_kw']:.1f},{hp['p_dc_max_kw']:.1f}]kW, hv_bad={hp['hv_bad']}",
            flush=True,
        )
    sl = probe["slalom_55kmh"]
    print(
        f"  Slalom 55 km/h: max|beta|={sl['max_beta_deg']:.2f}deg (<3.2), max|wz|={sl['max_wz_rads']:.3f}, "
        f"rec={sl['rec_time_s']:.3f}s, spun={sl['spun']}, hv_bad={sl['hv_bad']}",
        flush=True,
    )
    au = probe["solver_audit"]
    hv_5m_bad = sum(v["hv_bad"] for v in probe["hv_5maps"].values())
    print(
        f"  Solver & HV Audit: mean={au['step_time_mean_ms']:.3f}ms, p99={au['step_time_p99_ms']:.3f}ms (<1.5ms), "
        f"P_dc=[{au['p_dc_min_kw']:.2f}, {au['p_dc_max_kw']:.2f}]kW, viol_hi={au['hv_viol_hi_80kw']}, viol_lo={au['hv_viol_lo_25kw']}, 5map_hv_bad={hv_5m_bad}",
        flush=True,
    )


from concurrent.futures import ThreadPoolExecutor, as_completed


def run_ablation_suite(cases: list[tuple[str, int]], cfg_path: str | None = None, workers: int = 3) -> dict:
    stat_or = RESULTS_DIR / "stationary_oracle_report.json"
    hp_or = RESULTS_DIR / "hairpin_ocp_report.json"
    results: dict[str, dict] = {}

    def _run_one(label: str, mask: int) -> tuple[str, dict]:
        print(f"[ablation] Starting {label} (mask={mask})...", flush=True)
        probe = run_probe(cfg_path=cfg_path, ablation_mask=mask)
        gap = compute_gap_summary(probe, stat_or, hp_or)
        print(f"[ablation] Finished {label} (wall={probe.get('wall_time_s')}s)", flush=True)
        return label, {"mask": mask, "probe": probe, "gap": gap}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_one, lbl, msk) for lbl, msk in cases]
        for fut in as_completed(futs):
            lbl, res = fut.result()
            results[lbl] = res

    # Print compact comparison table in original order
    print("\n" + "=" * 124, flush=True)
    print(
        f"{'Variant':12s} | {'FSG R5':>7s} {'R9.1':>7s} {'R15':>7s} {'R25':>7s} {'FSG%':>6s} | "
        f"{'TTC R5':>7s} {'R9.1':>7s} {'R15':>7s} {'R25':>7s} {'TTC%':>6s} | "
        f"{'HP_FSG':>7s} {'HP_TTC':>7s} | {'SlalB':>6s} {'p99ms':>6s} {'HVbad':>5s}",
        flush=True,
    )
    print("-" * 124, flush=True)
    for lbl, _ in cases:
        r = results[lbl]
        g = r["gap"]
        p = r["probe"]
        fsg = g["stationary_gap"]["FSG_REAL_TRACK"]
        ttc = g["stationary_gap"]["TTC_LAB_PEAK"]
        hp_fsg = p["hairpin_180"]["FSG_REAL_TRACK"]["t_total_s"]
        hp_ttc = p["hairpin_180"]["TTC_LAB_PEAK"]["t_total_s"]
        sl_b = p["slalom_55kmh"]["max_beta_deg"]
        p99 = p["solver_audit"]["step_time_p99_ms"]
        hv_bad = (
            p["solver_audit"]["hv_viol_hi_80kw"]
            + p["solver_audit"]["hv_viol_lo_25kw"]
            + sum(v["hv_bad"] for v in p["hv_5maps"].values())
        )
        print(
            f"{lbl:12s} | "
            f"{fsg['R_5.000']['pct_of_oracle'] or 0:6.2f}% "
            f"{fsg['R_9.125']['pct_of_oracle'] or 0:6.2f}% "
            f"{fsg['R_15.000']['pct_of_oracle'] or 0:6.2f}% "
            f"{fsg['R_25.000']['pct_of_oracle'] or 0:6.2f}% "
            f"{fsg.get('mean_pct_of_oracle', 0):5.2f}% | "
            f"{ttc['R_5.000']['pct_of_oracle'] or 0:6.2f}% "
            f"{ttc['R_9.125']['pct_of_oracle'] or 0:6.2f}% "
            f"{ttc['R_15.000']['pct_of_oracle'] or 0:6.2f}% "
            f"{ttc['R_25.000']['pct_of_oracle'] or 0:6.2f}% "
            f"{ttc.get('mean_pct_of_oracle', 0):5.2f}% | "
            f"{hp_fsg:6.3f}s {hp_ttc:6.3f}s | {sl_b:5.2f}d {p99:5.3f} {hv_bad:5d}",
            flush=True,
        )
    print("=" * 124, flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure TV 4WD Optimality Gap on 14-DOF Godot model.")
    parser.add_argument("--label", default="baseline", help="Label for this run")
    parser.add_argument("--config", default=None, help="Optional path to tv_4wd_config.json")
    parser.add_argument("--ablation-mask", type=int, default=0, help="Bitmask: 1=A, 2=B, 4=C, 8=D, 16=E")
    parser.add_argument("--sweep-phase1", action="store_true", help="Run Baseline + individual A,B,C,D,E sweep")
    parser.add_argument("--sweep-custom", default=None, help="Comma-separated label:mask pairs, e.g. A+B:3,A+B+C:7")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.sweep_phase1:
        cases = [("Baseline", 0), ("+A", 1), ("+B", 2), ("+C", 4), ("+D", 8), ("+E", 16)]
        res = run_ablation_suite(cases, cfg_path=args.config, workers=3)
        out_path = Path(args.out) if args.out else (RESULTS_DIR / "ablation_phase1.json")
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}", flush=True)
        return
    if args.sweep_custom:
        cases = []
        for item in args.sweep_custom.split(","):
            lbl, msk = item.split(":")
            cases.append((lbl.strip(), int(msk.strip())))
        res = run_ablation_suite(cases, cfg_path=args.config, workers=min(3, len(cases)))
        out_path = Path(args.out) if args.out else (RESULTS_DIR / "ablation_custom.json")
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}", flush=True)
        return

    probe = run_probe(cfg_path=args.config, ablation_mask=args.ablation_mask)
    gap = compute_gap_summary(
        probe,
        RESULTS_DIR / "stationary_oracle_report.json",
        RESULTS_DIR / "hairpin_ocp_report.json",
    )
    print_summary(args.label, probe, gap)

    out_path = Path(args.out) if args.out else (RESULTS_DIR / f"gap_probe_{args.label}.json")
    payload = {"label": args.label, "ablation_mask": args.ablation_mask, "probe": probe, "gap": gap}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
