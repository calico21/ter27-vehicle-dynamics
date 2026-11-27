#!/usr/bin/env python3
"""
Tire Model Validation Framework — Unified Styling & Annotation Core
Provides publication-grade styling, colorblind-friendly model palettes,
consistent typography, SI units, and the automated unobtrusive in-plot metrics box.
"""

import matplotlib.pyplot as plt
import numpy as np

# Unified Model Colors (constant across all plots)
COLOR_EXP = "#2c3e50"        # Charcoal slate for experimental scatter
COLOR_M1_BASE = "#e74c3c"    # Red for Baseline MF 6.2
COLOR_M2_OPT = "#2980b9"     # Deep Blue for Optimized MF 6.2
COLOR_M3_HYB = "#27ae60"     # Forest Green for Hybrid MF + PINN

STYLE_M1 = {"color": COLOR_M1_BASE, "linestyle": ":", "linewidth": 1.8, "alpha": 0.85}
STYLE_M2 = {"color": COLOR_M2_OPT, "linestyle": "--", "linewidth": 2.0, "alpha": 0.90}
STYLE_M3 = {"color": COLOR_M3_HYB, "linestyle": "-", "linewidth": 2.4, "alpha": 0.95}

# 5 Physical Load Colors
LOAD_COLORS = [
    "#1f77b4",  # Level 1 (~220 N) - steel blue
    "#2ca02c",  # Level 2 (~440 N) - green
    "#ff7f0e",  # Level 3 (~665 N) - orange
    "#d62728",  # Level 4 (~885 N) - red
    "#9467bd"   # Level 5 (~1110 N) - purple
]

LOAD_LEVELS_INFO = [
    {"name": "Level 1 (~220 N)", "target": 222.0, "bin": (100.0, 330.0), "color": LOAD_COLORS[0]},
    {"name": "Level 2 (~440 N)", "target": 442.0, "bin": (330.0, 550.0), "color": LOAD_COLORS[1]},
    {"name": "Level 3 (~665 N)", "target": 665.0, "bin": (550.0, 770.0), "color": LOAD_COLORS[2]},
    {"name": "Level 4 (~885 N)", "target": 887.0, "bin": (770.0, 990.0), "color": LOAD_COLORS[3]},
    {"name": "Level 5 (~1110 N)", "target": 1111.0, "bin": (990.0, 1400.0), "color": LOAD_COLORS[4]},
]


def apply_theme():
    """Applies a clean, readable scientific theme."""
    style_name = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    plt.style.use(style_name)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#bdc3c7"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#ecf0f1"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.6


def add_metrics_box(ax, metrics_dict, loc="auto", title="Validation Metrics", unit_force="N", unit_mom="Nm"):
    """
    Adds a compact, elegant, borderless or subtly transparent metrics annotation box.
    
    Args:
        ax: matplotlib axes object
        metrics_dict: dict with keys for models, e.g.:
            {
               "Baseline": {"rmse": 183.5, "mae": 132.1, "bias": -12.4, "r2": 0.985, "n": 52294},
               "Optimized": {"rmse": 124.0, "mae": 92.0, "bias": -30.7, "r2": 0.993, "n": 52294},
               "Hybrid": {"rmse": 120.1, "mae": 89.2, "bias": -8.1, "r2": 0.994, "n": 52294}
            }
        loc: "auto", "upper left", "upper right", "lower left", "lower right"
        title: header title
        unit_force: unit label for force metrics
    """
    if not metrics_dict:
        return

    # Extract sample count from first model if available
    sample_count = None
    for m in metrics_dict.values():
        if "n" in m and m["n"] > 0:
            sample_count = m["n"]
            break

    # Format header
    lines = []
    if sample_count:
        if sample_count >= 1000:
            count_str = f"N={sample_count/1000.0:.1f}k"
        else:
            count_str = f"N={sample_count}"
        lines.append(f"{title} ({count_str})")
    else:
        lines.append(title)

    lines.append(f"{'Model':<9} {'RMSE':>6} {'MAE':>6} {'Bias':>6} {'R²':>5}")
    lines.append("-" * 36)

    for name, m in metrics_dict.items():
        # Shorten name if needed
        short_name = name.replace("Model ", "M").replace("model", "M")
        if len(short_name) > 9:
            short_name = short_name[:9]
        rmse_val = m.get("rmse", 0.0)
        mae_val = m.get("mae", 0.0)
        bias_val = m.get("bias", 0.0)
        r2_val = m.get("r2", 0.0)
        lines.append(f"{short_name:<9} {rmse_val:>6.1f} {mae_val:>6.1f} {bias_val:>+6.1f} {r2_val:>5.3f}")

    text_block = "\n".join(lines)

    # Determine location if "auto"
    if loc == "auto":
        # Estimate quadrant density based on lines plotted on ax
        # Default to upper left or lower right depending on common tire curves (Fy typically positive at positive alpha)
        # Check xlim and ylim
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if xlim[0] < 0 and ylim[0] < 0:
            # S-curve: top-left quadrant usually contains negative alpha or lower curve
            loc = "upper left"
        else:
            loc = "upper right"

    pos_map = {
        "upper left": (0.02, 0.98, "top", "left"),
        "upper right": (0.98, 0.98, "top", "right"),
        "lower left": (0.02, 0.02, "bottom", "left"),
        "lower right": (0.98, 0.02, "bottom", "right"),
    }
    x_pos, y_pos, va, ha = pos_map.get(loc, (0.02, 0.98, "top", "left"))

    ax.text(
        x_pos, y_pos, text_block,
        transform=ax.transAxes,
        fontsize=7.5,
        fontfamily="monospace",
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#ffffff",
            edgecolor="#bdc3c7",
            alpha=0.88,
            linewidth=0.8
        ),
        zorder=10
    )


def add_characterization_badge(ax, text="MODEL CHARACTERIZATION ONLY (NOT BENCH VALIDATED)"):
    """
    Adds a prominent, professional warning badge when a plot represents model characterization
    rather than direct experimental validation.
    """
    ax.text(
        0.5, 0.97, text,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color="#c0392b",
        horizontalalignment="center",
        verticalalignment="top",
        bbox=dict(
            boxstyle="square,pad=0.3",
            facecolor="#fdedec",
            edgecolor="#e74c3c",
            alpha=0.92,
            linewidth=1.0
        ),
        zorder=11
    )
