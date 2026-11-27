#!/usr/bin/env python3
"""
Estilo corporativo Tecnun eRacing, paleta de colores y exportación a 300 DPI.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# paleta tecnun eracing
C_REAL = "#1E293B"        # gris oscuro / pizarra (telemetría real can)
C_HORIZON = "#0066CC"     # azul eléctrico tecnun (horizonte 10s)
C_OPEN = "#E11D48"        # carmesí / rojo racing (lazo abierto)
C_TER26 = "#0066CC"       # azul ter26
C_TER27 = "#D97706"       # ámbar / naranja ter27
C_FL = "#0284C7"
C_FR = "#16A34A"
C_RL = "#D97706"
C_RR = "#DC2626"
C_ACCENT = "#059669"
C_MUTED = "#64748B"

DPI = 300


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
