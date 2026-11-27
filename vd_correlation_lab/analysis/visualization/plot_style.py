#!/usr/bin/env python3
"""
Shared MoTeC i2 / Motorsport Engineering Matplotlib style engine (300 DPI).

Design standards:
  - Dark-engineering figure canvas (#0D1117) and axes surface (#131922)
  - Crisp light foreground text (#E6EDF3) and subtle technical grid (#263040)
  - High-contrast, colorblind-safe motorsport signal palette (`PALETTE`)
  - Monospace font for KPI boxes, telemetry readouts, and tabular annotations
  - Pure ASCII text (zero emojis) and deterministic 300 DPI PNG export
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib as mpl

if mpl.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Colorblind-safe high-contrast motorsport palette (MoTeC i2 dark theme)
# ---------------------------------------------------------------------------
PALETTE: Dict[str, str] = {
    # Primary signal colors
    "cyan": "#38BDF8",      # Primary reference / telemetry / TV4 Auto
    "amber": "#F59E0B",     # Secondary / Open diff / warning threshold
    "emerald": "#10B981",   # Optimal / target / TV4 Eff / pass
    "crimson": "#F43F5E",   # Limit / error / violation / brake
    "violet": "#A78BFA",    # 3D kinematics / CasADi oracle / secondary trace
    "slate": "#94A3B8",     # Baseline / muted comparison / ghost
    "white": "#E6EDF3",     # Highlight / primary label / ground truth
    # Convenience aliases & additional accents
    "orange": "#FB923C",
    "blue": "#60A5FA",
    "teal": "#2DD4BF",
    "gold": "#FACC15",
    # Surface & chrome tokens
    "bg_fig": "#0D1117",
    "bg_ax": "#131922",
    "bg_box": "#161E29",
    "grid": "#263040",
    "grid_minor": "#1C2430",
    "spine": "#303B4C",
    "text": "#E6EDF3",
    "muted": "#9BA7B4",
}

COLOR_CYCLE: List[str] = [
    PALETTE["cyan"],
    PALETTE["amber"],
    PALETTE["emerald"],
    PALETTE["crimson"],
    PALETTE["violet"],
    PALETTE["orange"],
    PALETTE["teal"],
    PALETTE["slate"],
]

_KPI_LOC_MAP: Dict[str, Tuple[float, float, str, str]] = {
    "upper right": (0.98, 0.96, "right", "top"),
    "upper left": (0.02, 0.96, "left", "top"),
    "lower right": (0.98, 0.04, "right", "bottom"),
    "lower left": (0.02, 0.04, "left", "bottom"),
    "upper center": (0.50, 0.96, "center", "top"),
    "lower center": (0.50, 0.04, "center", "bottom"),
}


def apply_motec_style(dpi: int = 300) -> None:
    """Configure global Matplotlib rcParams for the MoTeC i2 dark-engineering theme."""
    mpl.rcParams.update(
        {
            # Resolution & figure surface
            "figure.dpi": 120,
            "savefig.dpi": dpi,
            "figure.facecolor": PALETTE["bg_fig"],
            "savefig.facecolor": PALETTE["bg_fig"],
            "savefig.edgecolor": PALETTE["bg_fig"],
            "figure.edgecolor": PALETTE["bg_fig"],
            # Axes surface & spines
            "axes.facecolor": PALETTE["bg_ax"],
            "axes.edgecolor": PALETTE["spine"],
            "axes.linewidth": 0.9,
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlepad": 8.0,
            "axes.labelsize": 9.2,
            "axes.labelpad": 5.0,
            "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
            "axes.grid": True,
            "axes.axisbelow": True,
            # Grid
            "grid.color": PALETTE["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.85,
            # Ticks
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            # Text & typography
            "text.color": PALETTE["text"],
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "mathtext.fontset": "dejavusans",
            # Lines & markers
            "lines.linewidth": 1.6,
            "lines.antialiased": True,
            "patch.edgecolor": PALETTE["spine"],
            # Legend
            "legend.facecolor": PALETTE["bg_box"],
            "legend.edgecolor": PALETTE["spine"],
            "legend.framealpha": 0.92,
            "legend.fontsize": 8.2,
            "legend.labelcolor": PALETTE["text"],
            "legend.borderpad": 0.45,
            "legend.handlelength": 1.8,
            # Suptitle
            "figure.titlesize": 12.5,
            "figure.titleweight": "bold",
        }
    )


def style_ax(
    ax: Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: bool = False,
    legend_loc: str = "best",
    legend_ncol: int = 1,
    grid_minor: bool = False,
) -> Axes:
    """Apply consistent MoTeC i2 axis styling, labels, grid, and optional legend."""
    ax.set_facecolor(PALETTE["bg_ax"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=10.5, fontweight="bold", loc="left", pad=7.0)
    if xlabel:
        ax.set_xlabel(xlabel, color=PALETTE["text"], fontsize=9.2)
    if ylabel:
        ax.set_ylabel(ylabel, color=PALETTE["text"], fontsize=9.2)

    ax.grid(True, which="major", color=PALETTE["grid"], linestyle="-", linewidth=0.65, alpha=0.85)
    if grid_minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", color=PALETTE["grid_minor"], linestyle=":", linewidth=0.45, alpha=0.65)

    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_color(PALETTE["grid"])
            spine.set_linewidth(0.6)
        else:
            spine.set_color(PALETTE["spine"])
            spine.set_linewidth(0.95)

    ax.tick_params(colors=PALETTE["muted"], labelsize=8.4)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(
                loc=legend_loc,
                ncol=legend_ncol,
                facecolor=PALETTE["bg_box"],
                edgecolor=PALETTE["spine"],
                framealpha=0.92,
                fontsize=8.1,
            )
            for txt in leg.get_texts():
                txt.set_color(PALETTE["text"])

    return ax


def add_kpi_box(
    ax: Axes,
    lines: Union[str, Sequence[str], Iterable[str]],
    loc: str = "upper right",
    title: Optional[str] = None,
    fontsize: float = 8.0,
    border_color: Optional[str] = None,
) -> mpl.text.Text:
    """Place a crisp monospace telemetry KPI summary box inside an axes."""
    if isinstance(lines, str):
        body = lines
    else:
        body = "\n".join(str(line) for line in lines)
    text = f"{title}\n{body}" if title else body

    x, y, ha, va = _KPI_LOC_MAP.get(loc, _KPI_LOC_MAP["upper right"])
    edge = border_color or PALETTE["cyan"]

    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontfamily="DejaVu Sans Mono",
        color=PALETTE["text"],
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.18",
            "facecolor": PALETTE["bg_box"],
            "edgecolor": edge,
            "linewidth": 0.9,
            "alpha": 0.92,
        },
        zorder=10,
    )


def add_suptitle(
    fig: Figure,
    title: str,
    subtitle: str = "",
    y: float = 0.98,
) -> None:
    """Add a Motorsport-styled figure suptitle and optional muted subtitle."""
    fig.suptitle(
        title,
        x=0.015,
        y=y,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=PALETTE["text"],
    )
    if subtitle:
        fig.text(
            0.015,
            y - 0.032,
            subtitle,
            ha="left",
            va="top",
            fontsize=8.6,
            fontfamily="DejaVu Sans Mono",
            color=PALETTE["muted"],
        )


def add_footer(
    fig: Figure,
    text: str = "TECNUN eRACING | VEHICLE DYNAMICS VALIDATION VAULT | ISO 8855",
) -> None:
    """Add a subtle monospace engineering footer along the bottom edge."""
    fig.text(
        0.99,
        0.008,
        text,
        ha="right",
        va="bottom",
        fontsize=7.2,
        fontfamily="DejaVu Sans Mono",
        color=PALETTE["muted"],
        alpha=0.80,
    )


def save_fig(
    fig: Figure,
    path: Union[str, Path],
    dpi: int = 300,
    close: bool = True,
    tight: bool = True,
) -> Path:
    """Save a figure at 300 DPI on the dark MoTeC canvas, creating parent dirs."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "dpi": dpi,
        "facecolor": fig.get_facecolor(),
        "edgecolor": "none",
    }
    if tight:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = 0.14
    fig.savefig(out_path, **save_kwargs)
    if close:
        plt.close(fig)
    return out_path
