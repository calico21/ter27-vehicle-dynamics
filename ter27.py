#!/home/alexr/project_gp_env/bin/python3
"""Unified CLI launcher for the TeR27 Vehicle Dynamics & Telemetry Workbench."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent
PREFERRED_PYTHON = Path("/home/alexr/project_gp_env/bin/python3")
FALLBACK_GODOT = Path("/home/alexr/.gemini/antigravity/bin/godot")
MIRENA_SIM_DIR = REPO_ROOT / "src" / "mirena_sim"


def resolve_python() -> str:
    """Return the project GP virtualenv Python executable if present."""
    if PREFERRED_PYTHON.exists():
        return str(PREFERRED_PYTHON)
    return sys.executable


def resolve_godot() -> str:
    """Return the Godot 4 executable path."""
    found = shutil.which("godot")
    if found:
        return found
    if FALLBACK_GODOT.exists():
        return str(FALLBACK_GODOT)
    raise FileNotFoundError(
        "Godot binary not found in PATH or /home/alexr/.gemini/antigravity/bin/godot"
    )


def run_cmd(cmd: List[str], cwd: Path = REPO_ROOT) -> int:
    """Execute a subprocess command with live unbuffered output."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[ter27] exec: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    return int(proc.returncode)


def cmd_sim(args: argparse.Namespace) -> int:
    """Launch the MirenaSim Godot 4 14-DOF simulator."""
    godot = resolve_godot()
    cmd = [godot]
    if args.headless:
        cmd.append("--headless")
    cmd.extend(["--path", str(MIRENA_SIM_DIR)])
    if args.extra:
        cmd.extend(args.extra)
    return run_cmd(cmd)


def cmd_api(args: argparse.Namespace) -> int:
    """Start the FastAPI telemetry and vehicle dynamics analysis backend."""
    py = resolve_python()
    cmd = [
        py,
        "-u",
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")
    return run_cmd(cmd)


def cmd_skidpad(args: argparse.Namespace) -> int:
    """Run the Skidpad CasADi optimal control solver or closed-loop sub-5s benchmark."""
    py = resolve_python()
    if args.benchmark:
        script = (
            REPO_ROOT
            / "vd_correlation_lab"
            / "analysis"
            / "benchmarks"
            / "run_skidpad_sub5_vault.py"
        )
    else:
        script = (
            REPO_ROOT
            / "vd_correlation_lab"
            / "optimization"
            / "skidpad"
            / "optimize_skidpad_tv_casadi.py"
        )
    cmd = [py, "-u", str(script)]
    if args.extra:
        cmd.extend(args.extra)
    return run_cmd(cmd)


def cmd_lts(args: argparse.Namespace) -> int:
    """Run CasADi dynamic events LTS or full-lap OCP solver."""
    py = resolve_python()
    if args.mode == "full-lap":
        script = (
            REPO_ROOT
            / "vd_correlation_lab"
            / "optimization"
            / "lap_time_sim"
            / "solve_full_lap_ocp.py"
        )
        cmd = [py, "-u", str(script)]
        if args.optimize_setup:
            cmd.append("--optimize-setup")
        if args.compound:
            cmd.extend(["--compound", args.compound])
    else:
        script = (
            REPO_ROOT
            / "vd_correlation_lab"
            / "optimization"
            / "lap_time_sim"
            / "solve_dynamic_events_casadi.py"
        )
        cmd = [
            py,
            "-u",
            str(script),
            "--event",
            args.event,
            "--grip-profile",
            args.grip_profile,
        ]
        if args.compound:
            cmd.extend(["--compound", args.compound])
        if args.optimize_setup:
            cmd.append("--optimize-setup")
    if args.extra:
        cmd.extend(args.extra)
    return run_cmd(cmd)


def cmd_vault(args: argparse.Namespace) -> int:
    """Rebuild the Validation Vault catalog, INDEX.md, and HTML dashboard."""
    py = resolve_python()
    script = (
        REPO_ROOT
        / "vd_correlation_lab"
        / "analysis"
        / "validation_vault"
        / "build_validation_vault_index.py"
    )
    cmd = [py, "-u", str(script)]
    if args.extra:
        cmd.extend(args.extra)
    return run_cmd(cmd)


def cmd_test(args: argparse.Namespace) -> int:
    """Run Python pytest regression suite and optionally Godot headless tests."""
    rc = 0
    if not args.godot_only:
        py = resolve_python()
        pytest_cmd = [py, "-u", "-m", "pytest", "vd_correlation_lab/tests/", "-v"]
        if args.extra:
            pytest_cmd.extend(args.extra)
        rc = run_cmd(pytest_cmd)
        if rc != 0:
            return rc

    if args.godot or args.godot_only:
        godot = resolve_godot()
        godot_cmd = [
            godot,
            "--headless",
            "--path",
            str(MIRENA_SIM_DIR),
            "-s",
            "../../tests/run_tests.gd",
        ]
        rc = run_cmd(godot_cmd)
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ter27.py",
        description="Unified CLI launcher for TeR27 Vehicle Dynamics, MirenaSim, and Telemetry API.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("sim", help="Launch MirenaSim (Godot 4 14-DOF simulator).")
    p_sim.add_argument("--headless", action="store_true", help="Run Godot in headless mode.")
    p_sim.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to Godot.")
    p_sim.set_defaults(func=cmd_sim)

    p_api = sub.add_parser("api", help="Start the FastAPI telemetry and analysis server (uvicorn).")
    p_api.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    p_api.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    p_api.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")
    p_api.set_defaults(func=cmd_api)

    p_skid = sub.add_parser(
        "skidpad",
        help="Run Skidpad CasADi optimal control solver or closed-loop benchmark.",
    )
    p_skid.add_argument(
        "--benchmark",
        action="store_true",
        help="Run closed-loop 14-DOF Skidpad sub-5s vault benchmark instead of CasADi OCP solver.",
    )
    p_skid.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to script.")
    p_skid.set_defaults(func=cmd_skidpad)

    p_lts = sub.add_parser("lts", help="Run CasADi LTS (dynamic events or full-lap OCP).")
    p_lts.add_argument(
        "--mode",
        choices=["events", "full-lap"],
        default="events",
        help="Solver mode: 'events' (solve_dynamic_events_casadi.py) or 'full-lap' (solve_full_lap_ocp.py).",
    )
    p_lts.add_argument(
        "--event",
        choices=["skidpad", "accel", "all"],
        default="all",
        help="Dynamic event for --mode events (default: all).",
    )
    p_lts.add_argument(
        "--grip-profile",
        default="FSG_RUBBERED",
        help="Grip profile for --mode events (default: FSG_RUBBERED).",
    )
    p_lts.add_argument(
        "--compound",
        default=None,
        help="Tire compound override (e.g. R25B or R20).",
    )
    p_lts.add_argument(
        "--optimize-setup",
        action="store_true",
        help="Co-optimize 7 trackside mechanical setup parameters.",
    )
    p_lts.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to solver.")
    p_lts.set_defaults(func=cmd_lts)

    p_vault = sub.add_parser("vault", help="Rebuild Validation Vault catalog, INDEX.md, and HTML dashboard.")
    p_vault.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to builder.")
    p_vault.set_defaults(func=cmd_vault)

    p_test = sub.add_parser("test", help="Run pytest suite and optionally Godot headless tests.")
    p_test.add_argument(
        "--godot",
        action="store_true",
        help="Also run Godot headless regression suite (tests/run_tests.gd).",
    )
    p_test.add_argument(
        "--godot-only",
        action="store_true",
        help="Run only Godot headless regression suite.",
    )
    p_test.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to pytest.")
    p_test.set_defaults(func=cmd_test)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
