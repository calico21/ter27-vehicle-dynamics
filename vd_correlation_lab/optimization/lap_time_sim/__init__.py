"""MirenaSim Hybrid Lap Time Simulator (LTS) package for TeR27 4WD."""

from .track import Track, load_planning_cone_track, build_track_from_waypoints
from .racing_line import RacingLineResult, optimize_racing_line
from .mintime_ocp import optimize_mintime_racing_line, export_enriched_racing_line_json, verify_swept_path_4corners
from .ggv_extractor import GGVMap, extract_or_load_ggv
from .qss_solver import QSSResult, solve_qss
from .transient_lts import TransientLapResult, run_transient_lap, search_fastest_valid_lap
from .scoring import EnduranceProjection, compute_fs_scores, project_endurance_22km

__all__ = [
    "Track",
    "load_planning_cone_track",
    "build_track_from_waypoints",
    "RacingLineResult",
    "optimize_racing_line",
    "optimize_mintime_racing_line",
    "export_enriched_racing_line_json",
    "verify_swept_path_4corners",
    "GGVMap",
    "extract_or_load_ggv",
    "QSSResult",
    "solve_qss",
    "TransientLapResult",
    "run_transient_lap",
    "search_fastest_valid_lap",
    "EnduranceProjection",
    "compute_fs_scores",
    "project_endurance_22km",
]
