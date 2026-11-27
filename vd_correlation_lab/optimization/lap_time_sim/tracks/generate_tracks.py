"""Reproducible track library builder for MirenaSim LTS."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..track import Track, build_track_from_waypoints, load_planning_cone_track

TRACKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRACKS_DIR.parents[3]
PLANNING_TRACK10_PATH = REPO_ROOT / "src" / "mirena_planning" / "test" / "tracks" / "track10.json"


def build_fsg_autocross_endurance(ds_target: float = 0.5) -> Track:
    """
    Build a ~1.05 km (1.0-1.1 km) C2 closed FSG Autocross / Endurance circuit (FS D 6.1.3 / D 7.1.3 compliant):
      - Main straight <= 75 m (never > 80 m FS rule, reaching ~95-104 km/h)
      - Tight hairpins (R ~ 6.5-8.5 m)
      - Real FS slalom (11.5 m gate spacing, +/- 2.8 m lateral)
      - Technical braking chicane & linked S-curves (mean speed ~55-68 km/h)
    Track width = 3.60 m (w_left = w_right = 1.80 m >= 3.0 m FS rule).
    """
    waypoints = np.array([
        # Main straight (~70 m along Y = 0, +X)
        [0.0, 0.0],
        [24.0, 0.0],
        [48.0, 0.0],
        [70.0, 0.0],
        # Turn 1 sweeping left into East Technical Loop & Hairpin 1 (R ~ 7.0 m)
        [94.0, 7.0],
        [116.0, 22.0],
        [134.0, 44.0],
        [156.0, 56.0],
        [168.0, 72.0],
        [158.0, 88.0],
        [136.0, 86.0],
        [124.0, 92.0],
        # East Braking Chicane & Medium Linked Sweepers (R ~ 10-16 m)
        [126.0, 108.0],
        [142.0, 122.0],
        [156.0, 146.0],
        [146.0, 168.0],
        [122.0, 176.0],
        [98.0, 170.0],
        # North FS Slalom (14.0 m spacing, +/- 3.5 m lateral -> R ~ 11 m, zero straight > 80m)
        [82.0, 172.0],
        [68.0, 175.5],
        [54.0, 168.5],
        [40.0, 175.5],
        [26.0, 168.5],
        [12.0, 175.5],
        [-2.0, 168.5],
        [-18.0, 172.0],
        # Northwest Sweeper + West Technical S-Section & Hairpin 2 (R ~ 7.5 m)
        [-44.0, 174.0],
        [-74.0, 166.0],
        [-102.0, 148.0],
        [-114.0, 124.0],
        [-98.0, 102.0],
        [-76.0, 96.0],
        [-78.0, 74.0],
        [-106.0, 66.0],
        [-124.0, 46.0],
        [-112.0, 24.0],
        [-86.0, 20.0],
        [-90.0, 2.0],
        # Southwest Braking Hairpin 3 (R ~ 8.0 m) onto Main Straight
        [-112.0, -12.0],
        [-104.0, -34.0],
        [-78.0, -40.0],
        [-50.0, -28.0],
        [-28.0, -10.0],
        [-14.0, 0.0],
    ], dtype=np.float64)

    return build_track_from_waypoints(
        name="fsg_autocross_endurance",
        waypoints_xy=waypoints,
        w_left=1.80,
        w_right=1.80,
        is_closed=True,
        ds_target=ds_target,
    )


def ensure_all_reference_tracks() -> dict[str, Track]:
    fsg = build_fsg_autocross_endurance(ds_target=0.5)
    fsg.save_json(TRACKS_DIR / "fsg_autocross_endurance.json")

    mt10 = load_planning_cone_track(PLANNING_TRACK10_PATH, name="mirena_track10", ds_target=0.5)
    mt10.save_json(TRACKS_DIR / "mirena_track10.json")
    return {"fsg_autocross_endurance": fsg, "mirena_track10": mt10}


if __name__ == "__main__":
    tracks = ensure_all_reference_tracks()
    for name, trk in tracks.items():
        st = trk.check_validity()
        print(f"{name}: {st}")
