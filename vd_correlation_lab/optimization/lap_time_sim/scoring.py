"""Official Formula Student (FSG-style) scoring & 22 km Endurance projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EnduranceProjection:
    """22 km Endurance projection metrics from single-lap / multi-lap LTS run."""

    label: str
    track_length_m: float
    t_lap_s: float
    n_laps_22km: float
    t_22km_s: float
    e_net_lap_kj: float
    e_regen_lap_kj: float
    e_net_22km_kwh: float
    e_regen_22km_kwh: float
    kwh_per_km: float
    regen_pct: float
    e_t2_index: float
    soc_used_pct_of_7_2kwh: float


def project_endurance_22km(
    label: str,
    track_length_m: float,
    t_lap_s: float,
    e_net_lap_kj: float,
    e_regen_lap_kj: float,
    battery_capacity_kwh: float = 7.2,
) -> EnduranceProjection:
    n_laps = 22000.0 / max(track_length_m, 1.0)
    t_22km = t_lap_s * n_laps
    e_net_kwh = (e_net_lap_kj / 3600.0) * n_laps
    e_reg_kwh = (e_regen_lap_kj / 3600.0) * n_laps
    e_pos_lap = max(e_net_lap_kj + e_regen_lap_kj, 1.0e-6)
    regen_pct = 100.0 * max(e_regen_lap_kj, 0.0) / e_pos_lap
    return EnduranceProjection(
        label=label,
        track_length_m=float(track_length_m),
        t_lap_s=float(t_lap_s),
        n_laps_22km=float(n_laps),
        t_22km_s=float(t_22km),
        e_net_lap_kj=float(e_net_lap_kj),
        e_regen_lap_kj=float(e_regen_lap_kj),
        e_net_22km_kwh=float(e_net_kwh),
        e_regen_22km_kwh=float(e_reg_kwh),
        kwh_per_km=float(e_net_kwh / 22.0),
        regen_pct=float(regen_pct),
        e_t2_index=float(e_net_lap_kj * (t_lap_s ** 2)),
        soc_used_pct_of_7_2kwh=100.0 * e_net_kwh / max(battery_capacity_kwh, 0.1),
    )


def compute_fs_scores(projections: List[EnduranceProjection]) -> List[Dict[str, object]]:
    """
    Compute relative FSG scores across a list of configurations:
      - Autocross (100 pts max): 4.5 + 95.5 * ((1.45 * t_min / t) - 1) / 0.45  (clamped [4.5, 100] when valid)
      - Endurance (250 pts max): 25 + 225 * ((1.333 * t_min / t) - 1) / 0.333
      - Efficiency (100 pts max): FS EV E * t^2 normalized relative to best eligible (t <= 1.333 * t_min)
    """
    if not projections:
        return []
    t_min = min(p.t_lap_s for p in projections)
    eligible = [p for p in projections if p.t_lap_s <= 1.333 * t_min and p.e_net_lap_kj > 0.0]
    et2_best = min((p.e_t2_index for p in eligible), default=min(p.e_t2_index for p in projections))
    et2_worst = max((p.e_t2_index for p in eligible), default=max(p.e_t2_index for p in projections))

    rows: List[Dict[str, object]] = []
    for p in projections:
        t_max_auto = 1.45 * t_min
        if p.t_lap_s >= t_max_auto:
            pts_auto = 4.5
        else:
            pts_auto = 4.5 + 95.5 * ((t_max_auto / p.t_lap_s) - 1.0) / 0.45

        t_max_end = 1.333 * t_min
        if p.t_lap_s >= t_max_end:
            pts_end = 25.0
            pts_eff = 0.0
        else:
            pts_end = 25.0 + 225.0 * ((t_max_end / p.t_lap_s) - 1.0) / 0.333
            # FSG efficiency factor EF = (E_min * t_min^2) / (E * t^2) relative scale
            if et2_worst > et2_best + 1e-6:
                ef_cur = et2_best / max(p.e_t2_index, 1e-6)
                ef_min = et2_best / max(et2_worst, 1e-6)
                pts_eff = 100.0 * max(0.0, min(1.0, (ef_cur - 0.5 * ef_min) / max(1.0 - 0.5 * ef_min, 1e-6)))
            else:
                pts_eff = 100.0

        rows.append({
            "label": p.label,
            "t_lap_s": p.t_lap_s,
            "delta_t_s": p.t_lap_s - t_min,
            "t_22km_s": p.t_22km_s,
            "e_net_lap_kj": p.e_net_lap_kj,
            "e_regen_lap_kj": p.e_regen_lap_kj,
            "e_net_22km_kwh": p.e_net_22km_kwh,
            "kwh_per_km": p.kwh_per_km,
            "regen_pct": p.regen_pct,
            "e_t2_index": p.e_t2_index,
            "e_t2_rel_best": p.e_t2_index / max(et2_best, 1e-6),
            "soc_used_pct": p.soc_used_pct_of_7_2kwh,
            "pts_autocross_rel": float(min(100.0, max(0.0, pts_auto))),
            "pts_endurance_rel": float(min(250.0, max(0.0, pts_end))),
            "pts_efficiency_rel": float(min(100.0, max(0.0, pts_eff))),
            "pts_total_dynamic_rel": float(min(450.0, pts_auto + pts_end + pts_eff)),
        })
    return rows
