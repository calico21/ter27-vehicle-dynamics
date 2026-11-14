#!/usr/bin/env python3
"""Multi-objective Torque Vectoring (TV) parameter optimizer for Mirena (TeR27).

Discovers circuits under src/mirena_sim/TrackFiles/*.json, runs parallel closed-loop
14-DOF simulations via headless Godot evaluations of test_tv_characterization.gd,
extracts the non-dominated Pareto front across 5 dynamics/efficiency objectives,
and recommends optimal parameter sets for Endurance, Autocross, and Skidpad.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _godot_runner import extract_result_json, find_godot_binary


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT_DIR = REPO_ROOT / "src" / "mirena_sim"
DEFAULT_TRACKS_DIR = DEFAULT_PROJECT_DIR / "TrackFiles"
TEST_SCRIPT_RES = "res://tests/test_tv_characterization.gd"
RESULT_PREFIX = "TV_OPT_RESULT:"

OBJECTIVES: tuple[str, ...] = (
    "sector_time_s",
    "wz_rmse_rads",
    "efficiency_kwh_km",
    "peak_beta_deg",
    "final_tire_temp_c",
)

# Physical search bounds for ECU TV V3 parameters:
# (GP_K_US [rad/(m/s)^2], GP_TV_MAX_MZ [Nm], GP_TC_KP [Nm/slip], min_fx_reserve [-])
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "k_us": (0.003, 0.025),
    "max_mz": (150.0, 1200.0),
    "tc_kp": (12.0, 32.0),
    "fx_res": (0.04, 0.16),
}

BASELINE_PARAMS: dict[str, float] = {
    "k_us": 0.015,
    "max_mz": 500.0,
    "tc_kp": 23.113,
    "fx_res": 0.08,
}

EVENT_WEIGHTS: dict[str, dict[str, float]] = {
    "ENDURANCE": {
        "sector_time_s": 0.15,
        "wz_rmse_rads": 0.05,
        "efficiency_kwh_km": 0.35,
        "peak_beta_deg": 0.30,
        "final_tire_temp_c": 0.15,
    },
    "AUTOCROSS": {
        "sector_time_s": 0.45,
        "wz_rmse_rads": 0.30,
        "efficiency_kwh_km": 0.05,
        "peak_beta_deg": 0.15,
        "final_tire_temp_c": 0.05,
    },
    "SKIDPAD": {
        "sector_time_s": 0.30,
        "wz_rmse_rads": 0.55,
        "efficiency_kwh_km": 0.00,
        "peak_beta_deg": 0.10,
        "final_tire_temp_c": 0.05,
    },
}


@dataclass(frozen=True)
class EvalJob:
    index: int
    params: dict[str, float]
    tracks: list[str]
    project_dir: str
    godot_bin: str
    map_id: int
    timeout_s: float


def resolve_godot_bin(requested: str | None) -> str:
    """Resolve the Godot binary path from CLI flag, PATH, or standard install."""
    if requested:
        return requested
    found = shutil.which("godot")
    if found:
        return found
    fallback = Path.home() / ".gemini" / "antigravity" / "bin" / "godot"
    if fallback.exists():
        return str(fallback)
    return "godot"


def discover_tracks(tracks_dir: Path) -> list[str]:
    """Discover all *.json circuit names in TrackFiles/ deterministically."""
    if not tracks_dir.is_dir():
        return []
    files = sorted(p for p in tracks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    return [p.stem for p in files]


def _halton(index: int, base: int) -> float:
    """Compute 1-based Halton quasi-random coordinate in (0, 1)."""
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= float(base)
        r += f * float(i % base)
        i //= base
    return r


def generate_parameter_samples(n_samples: int) -> list[dict[str, float]]:
    """Build deterministic baseline + 4D Halton structured sweep over TV bounds."""
    if n_samples <= 0:
        return []
    samples: list[dict[str, float]] = [dict(BASELINE_PARAMS)]
    if n_samples == 1:
        return samples

    primes = (2, 3, 5, 7)
    keys = ("k_us", "max_mz", "tc_kp", "fx_res")
    precision = {"k_us": 5, "max_mz": 2, "tc_kp": 3, "fx_res": 4}

    for idx in range(1, n_samples):
        cand: dict[str, float] = {}
        for key, base in zip(keys, primes):
            lo, hi = PARAM_BOUNDS[key]
            u = _halton(idx, base)
            cand[key] = round(lo + u * (hi - lo), precision[key])
        samples.append(cand)
    return samples


def _extract_result_json(stdout_text: str) -> dict[str, Any] | None:
    """Parse TV_OPT_RESULT:{...} JSON line emitted by test_tv_characterization.gd."""
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(RESULT_PREFIX):
            payload = stripped[len(RESULT_PREFIX) :].strip()
            return json.loads(payload)
    return None


def _run_single_evaluation(job: EvalJob) -> dict[str, Any]:
    """Worker entrypoint executed in ProcessPoolExecutor."""
    t0 = time.perf_counter()
    p = job.params
    cmd = [
        job.godot_bin,
        "--headless",
        "--path",
        job.project_dir,
        "-s",
        TEST_SCRIPT_RES,
        "--",
        "--eval-params",
        f"k_us={p['k_us']:.6f}",
        f"max_mz={p['max_mz']:.3f}",
        f"tc_kp={p['tc_kp']:.4f}",
        f"fx_res={p['fx_res']:.5f}",
        f"--map={job.map_id}",
    ]
    if job.tracks:
        cmd.append("--tracks=" + ",".join(job.tracks))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=job.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "index": job.index,
            "ok": False,
            "error": f"timeout after {job.timeout_s:.1f}s",
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "params": p,
            "metrics": {},
            "tracks": {},
        }
    except OSError as exc:
        return {
            "index": job.index,
            "ok": False,
            "error": f"launch failed: {exc}",
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "params": p,
            "metrics": {},
            "tracks": {},
        }

    elapsed_s = round(time.perf_counter() - t0, 3)
    parsed = _extract_result_json(proc.stdout)
    if parsed is None:
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        return {
            "index": job.index,
            "ok": False,
            "error": f"missing {RESULT_PREFIX} (rc={proc.returncode}): {tail}",
            "elapsed_s": elapsed_s,
            "params": p,
            "metrics": {},
            "tracks": {},
        }

    metrics = parsed.get("metrics", {})
    all_finite = all(
        math.isfinite(float(metrics.get(obj, math.nan))) for obj in OBJECTIVES
    )
    return {
        "index": job.index,
        "ok": bool(parsed.get("ok", False)) and all_finite,
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "params": parsed.get("params", p),
        "metrics": metrics,
        "tracks": parsed.get("tracks", {}),
    }


def _dominates(cand_a: dict[str, Any], cand_b: dict[str, Any], objectives: tuple[str, ...]) -> bool:
    """Return True if cand_a Pareto-dominates cand_b (all objectives minimized)."""
    ma = cand_a.get("metrics", {})
    mb = cand_b.get("metrics", {})
    strictly_better = False
    for obj in objectives:
        va = float(ma.get(obj, math.inf))
        vb = float(mb.get(obj, math.inf))
        if va > vb + 1e-12:
            return False
        if va < vb - 1e-12:
            strictly_better = True
    return strictly_better


def compute_pareto_front(
    candidates: list[dict[str, Any]],
    objectives: tuple[str, ...] = OBJECTIVES,
) -> list[dict[str, Any]]:
    """Filter candidates to the non-dominated Pareto front."""
    front: list[dict[str, Any]] = []
    for i, ci in enumerate(candidates):
        dominated = False
        for j, cj in enumerate(candidates):
            if i != j and _dominates(cj, ci, objectives):
                dominated = True
                break
        if not dominated:
            front.append(ci)
    front.sort(key=lambda r: float(r["metrics"].get("sector_time_s", math.inf)))
    return front


def _pick_metric_source(entry: dict[str, Any], event_name: str) -> dict[str, Any]:
    """Use skidpad-specific metrics when recommending SKIDPAD if available."""
    if event_name == "SKIDPAD":
        tracks = entry.get("tracks", {})
        for tid, tdata in tracks.items():
            if "skidpad" in tid.lower() and isinstance(tdata, dict):
                return tdata
    return entry.get("metrics", {})


def select_event_recommendations(
    pareto_front: list[dict[str, Any]],
    feasible_pool: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Select best Pareto-optimal operating point for Endurance, Autocross, and Skidpad."""
    pool = pareto_front if pareto_front else feasible_pool
    if not pool:
        return {}

    # Physical minimum normalization spans prevent numerical noise from dominating weights
    span_floors: dict[str, float] = {
        "sector_time_s": 0.05,
        "wz_rmse_rads": 0.02,
        "efficiency_kwh_km": 1e-4,
        "peak_beta_deg": 0.10,
        "final_tire_temp_c": 0.01,
    }

    recs: dict[str, dict[str, Any]] = {}
    for event_name, weights in EVENT_WEIGHTS.items():
        # Normalize each objective across the reference pool for balanced weighting.
        bounds: dict[str, tuple[float, float]] = {}
        for obj in OBJECTIVES:
            vals = [
                float(_pick_metric_source(c, event_name).get(obj, math.inf))
                for c in feasible_pool
                if math.isfinite(float(_pick_metric_source(c, event_name).get(obj, math.nan)))
            ]
            bounds[obj] = (min(vals), max(vals)) if vals else (0.0, 1.0)

        best_entry: dict[str, Any] | None = None
        best_score = math.inf
        for cand in pool:
            src = _pick_metric_source(cand, event_name)
            score = 0.0
            for obj, w in weights.items():
                if w <= 0.0:
                    continue
                val = float(src.get(obj, math.inf))
                lo, hi = bounds[obj]
                span = max(hi - lo, span_floors.get(obj, 1e-6))
                norm = (val - lo) / span
                score += w * norm
            if score < best_score:
                best_score = score
                best_entry = cand

        if best_entry is not None:
            recs[event_name] = {
                "score": round(best_score, 5),
                "sample_index": best_entry["index"],
                "params": best_entry["params"],
                "metrics": best_entry["metrics"],
                "event_metrics": _pick_metric_source(best_entry, event_name),
            }
    return recs


def format_markdown_report(summary: dict[str, Any]) -> str:
    """Render structured Markdown optimization summary."""
    lines: list[str] = [
        "# Mirena Torque Vectoring Multi-Objective Optimization Report",
        "",
        f"- **Discovered Tracks**: {', '.join(summary['tracks']) if summary['tracks'] else 'none'}",
        f"- **Evaluated Samples**: {summary['evaluated_count']} "
        f"(Feasible: {summary['feasible_count']}, Pareto Front: {len(summary['pareto_front'])})",
        f"- **Total Wall Time**: {summary['wall_time_s']:.2f} s",
        "",
        "## Recommended Optimal Maps by Event",
        "",
        "| Event | GP_K_US | GP_TV_MAX_MZ (Nm) | GP_TC_KP | min_fx_reserve | Time (s) | wz RMSE (rad/s) | Peak Beta (deg) | kWh/km | T_surf (C) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for event_name in ("ENDURANCE", "AUTOCROSS", "SKIDPAD"):
        rec = summary["recommendations"].get(event_name)
        if not rec:
            continue
        p = rec["params"]
        m = rec["event_metrics"]
        lines.append(
            f"| {event_name} | {p['k_us']:.5f} | {p['max_mz']:.1f} | {p['tc_kp']:.2f} | {p['fx_res']:.4f} | "
            f"{m.get('sector_time_s', float('nan')):.3f} | {m.get('wz_rmse_rads', float('nan')):.4f} | "
            f"{m.get('peak_beta_deg', float('nan')):.2f} | {m.get('efficiency_kwh_km', float('nan')):.3f} | "
            f"{m.get('final_tire_temp_c', float('nan')):.2f} |"
        )

    lines.extend([
        "",
        "## Non-Dominated Pareto Front",
        "",
        "| # | k_us | max_mz | tc_kp | fx_res | Time (s) | wz RMSE | Peak Beta (deg) | kWh/km | T_surf (C) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["pareto_front"]:
        p = row["params"]
        m = row["metrics"]
        lines.append(
            f"| {row['index']} | {p['k_us']:.5f} | {p['max_mz']:.1f} | {p['tc_kp']:.2f} | {p['fx_res']:.4f} | "
            f"{m['sector_time_s']:.3f} | {m['wz_rmse_rads']:.4f} | {m['peak_beta_deg']:.2f} | "
            f"{m['efficiency_kwh_km']:.3f} | {m['final_tire_temp_c']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def print_stdout_report(summary: dict[str, Any]) -> None:
    """Print concise tables of sweep results, Pareto front, and recommendations."""
    print("\n" + "=" * 96)
    print("  MIRENA DYNAMICS (TeR27) — MULTI-OBJECTIVE TV PARAMETER OPTIMIZATION")
    print("=" * 96)
    print(f"Tracks evaluated : {', '.join(summary['tracks'])}")
    print(
        f"Samples          : {summary['evaluated_count']} total | "
        f"{summary['feasible_count']} feasible | {len(summary['pareto_front'])} Pareto non-dominated"
    )
    print(f"Wall time        : {summary['wall_time_s']:.2f} s\n")

    print("[PARETO FRONT (NON-DOMINATED)]")
    print(
        "  Id |    k_us |  max_mz | tc_kp | fx_res |  Time(s) |  wzRMSE | PkBeta | kWh/km | T_surf"
    )
    print("  " + "-" * 90)
    for row in summary["pareto_front"]:
        p = row["params"]
        m = row["metrics"]
        print(
            f"  {row['index']:2d} | {p['k_us']:7.5f} | {p['max_mz']:7.1f} | {p['tc_kp']:5.2f} | {p['fx_res']:6.4f} | "
            f"{m['sector_time_s']:8.3f} | {m['wz_rmse_rads']:7.4f} | {m['peak_beta_deg']:5.2f}d | "
            f"{m['efficiency_kwh_km']:6.3f} | {m['final_tire_temp_c']:5.2f}C"
        )

    print("\n[RECOMMENDED OPTIMAL TV PARAMETERS BY EVENT]")
    for event_name in ("ENDURANCE", "AUTOCROSS", "SKIDPAD"):
        rec = summary["recommendations"].get(event_name)
        if not rec:
            print(f"  {event_name:10s}: <no feasible solution>")
            continue
        p = rec["params"]
        m = rec["event_metrics"]
        print(
            f"  {event_name:10s} (sample #{rec['sample_index']:2d}): "
            f"GP_K_US={p['k_us']:.5f}, GP_TV_MAX_MZ={p['max_mz']:.1f}, "
            f"GP_TC_KP={p['tc_kp']:.2f}, min_fx_reserve={p['fx_res']:.4f}  |  "
            f"t={m.get('sector_time_s', float('nan')):.3f}s, "
            f"wzRMSE={m.get('wz_rmse_rads', float('nan')):.4f}, "
            f"beta={m.get('peak_beta_deg', float('nan')):.2f}deg, "
            f"eff={m.get('efficiency_kwh_km', float('nan')):.3f}kWh/km, "
            f"T={m.get('final_tire_temp_c', float('nan')):.2f}C"
        )
    print()


def write_optional_output(output_arg: str | None, summary: dict[str, Any]) -> None:
    """Write JSON/Markdown report only if explicitly requested via --output."""
    if not output_arg:
        return
    out_path = Path(output_arg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".md":
        out_path.write_text(format_markdown_report(summary), encoding="utf-8")
        print(f"Saved Markdown report to {out_path}")
    elif suffix == ".json":
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {out_path}")
    else:
        json_path = out_path.with_suffix(".json")
        md_path = out_path.with_suffix(".md")
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        md_path.write_text(format_markdown_report(summary), encoding="utf-8")
        print(f"Saved JSON report to {json_path} and Markdown report to {md_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel multi-objective Torque Vectoring optimizer via headless Godot."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=24,
        help="Number of parameter configurations to evaluate (default: 24).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 2)),
        help="Parallel Godot worker processes (default: min(4, cpu_count)).",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default="",
        help="Comma-separated TrackFiles stems to evaluate (default: auto-discover all *.json).",
    )
    parser.add_argument(
        "--map-id",
        type=int,
        default=2,
        help="Base ECU TV map ID before applying custom parameters (default: 2 = AUTOCROSS_NEUTRAL).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-evaluation Godot subprocess timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--godot-bin",
        type=str,
        default="",
        help="Optional explicit path to godot binary.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional report output path (.json, .md, or stem for both).",
    )
    parser.add_argument(
        "--export-json-dir",
        type=str,
        default="",
        help="Optional directory to export recommended TV maps as tv_maps/*.json files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    godot_bin = resolve_godot_bin(args.godot_bin or None)

    discovered = discover_tracks(DEFAULT_TRACKS_DIR)
    if args.tracks.strip():
        tracks = [t.strip().lower() for t in args.tracks.split(",") if t.strip()]
    else:
        tracks = discovered

    if not tracks:
        print(f"ERROR: No tracks discovered in {DEFAULT_TRACKS_DIR}", file=sys.stderr)
        return 2

    candidates = generate_parameter_samples(args.samples)
    jobs = [
        EvalJob(
            index=i,
            params=params,
            tracks=tracks,
            project_dir=str(DEFAULT_PROJECT_DIR),
            godot_bin=godot_bin,
            map_id=args.map_id,
            timeout_s=args.timeout,
        )
        for i, params in enumerate(candidates)
    ]

    t_wall0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    max_workers = max(1, min(args.workers, len(jobs)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_run_single_evaluation, job): job.index for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            res = future.result()
            results.append(res)
            p = res["params"]
            if res["ok"]:
                m = res["metrics"]
                print(
                    f"[eval {res['index']:2d}/{len(jobs)-1}] OK ({res['elapsed_s']:.1f}s) "
                    f"k_us={p['k_us']:.4f} mz={p['max_mz']:.0f} kp={p['tc_kp']:.1f} res={p['fx_res']:.3f} -> "
                    f"t={m['sector_time_s']:.3f}s wzRMSE={m['wz_rmse_rads']:.4f} "
                    f"beta={m['peak_beta_deg']:.2f}deg eff={m['efficiency_kwh_km']:.3f} T={m['final_tire_temp_c']:.2f}C",
                    flush=True,
                )
            else:
                print(
                    f"[eval {res['index']:2d}/{len(jobs)-1}] FAIL ({res['elapsed_s']:.1f}s): {res.get('error', 'incomplete')}",
                    flush=True,
                )

    results.sort(key=lambda r: r["index"])
    wall_time_s = time.perf_counter() - t_wall0

    feasible = [r for r in results if r["ok"]]
    pareto_pool = feasible if feasible else [r for r in results if r.get("metrics")]
    pareto_front = compute_pareto_front(pareto_pool, OBJECTIVES)
    recommendations = select_event_recommendations(pareto_front, pareto_pool)

    summary: dict[str, Any] = {
        "tracks": tracks,
        "evaluated_count": len(results),
        "feasible_count": len(feasible),
        "wall_time_s": round(wall_time_s, 3),
        "pareto_front": pareto_front,
        "recommendations": recommendations,
        "all_results": results,
    }

    print_stdout_report(summary)
    write_optional_output(args.output.strip() or None, summary)
    export_dir = args.export_json_dir.strip()
    if export_dir and recommendations:
        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for event_name, rec in recommendations.items():
            p = rec.get("params", {})
            payload = {
                "name": f"Pareto {event_name.title()}",
                "description": f"Auto-tuned Pareto recommendation for {event_name}",
                "author": "optimize_tv_maps.py",
                "params": {
                    "k_us": p.get("k_us", 0.011),
                    "max_mz": p.get("max_mz", 750.0),
                    "min_fx_reserve": p.get("min_fx_reserve", 0.06),
                    "tc_kp": p.get("tc_kp", 22.5),
                    "k_beta": p.get("k_beta", 4800.0),
                    "throttle_mz_cut": p.get("throttle_mz_cut", 0.45),
                    "front_drive_bias": p.get("front_drive_bias", 0.35),
                    "can_latency_ms": p.get("can_latency_ms", 12.0),
                },
            }
            out_file = out_dir / f"pareto_{event_name.lower().replace(' ', '_')}.json"
            out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
