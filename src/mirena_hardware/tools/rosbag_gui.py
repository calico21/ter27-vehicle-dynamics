#!/usr/bin/env python3
"""
rosbag_gui.py — Interfaz grafica para rosbag_analyzer.py

    source /opt/ros/jazzy/setup.bash
    source ~/ws/install/setup.bash
    python3 rosbag_gui.py [/ruta/al/bag | /ruta/al/fichero.mcap]

Requiere bag_analayzer.py y bag_replay.py en la misma carpeta, y tkinter
(sudo apt install python3-tk).

  * Ventana principal: lista de topics, informe de texto y resumen general.
  * Cada grafica se abre en su propia ventana con barra de zoom/pan/guardar
    (y rotacion con el raton en las figuras 3D).
  * Seccion "Solo IMU" y "Solo GPS": dashboard, selector de canales y 3D.
  * "Solo IMU + GPS" al cargar: lee unicamente esos topics (carga rapida).
  * Ventana temporal (inicio/fin en segundos) aplicada a todas las graficas.
  * Pestana "Reproduccion + Kalman": reproduce el bag, arranca el nodo y
    dibuja en vivo el sensor solo frente al EKF, sin abrir otra ventana.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
import os
import queue
import shlex
import sys
import threading
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    raise SystemExit("\nFalta tkinter:  sudo apt install python3-tk\n")

import bag_analayzer as A

A.setup_style(True)          # backend Agg: las figuras se incrustan a mano en Tk
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg, NavigationToolbar2Tk)

try:                         # el modulo de reproduccion es opcional
    import bag_replay as R  # noqa: E402
except Exception:            # noqa: BLE001
    R = None


# --------------------------------------------------------------------------- #
#  Figuras "solo canales elegidos" (IMU y GPS)
# --------------------------------------------------------------------------- #

IMU_GROUPS = [
    ("Aceleración", [("acc_x", "x", True), ("acc_y", "y", True),
                     ("acc_z", "z", True), ("acc_norm", "|a|", False)]),
    ("Velocidad angular", [("gyr_x", "x", True), ("gyr_y", "y", True),
                           ("gyr_z", "z", True), ("gyr_norm", "|ω|", False)]),
    ("Orientación", [("rpy_roll", "roll", False), ("rpy_pitch", "pitch", False),
                     ("rpy_yaw", "yaw", False)]),
    ("Espectros", [("psd_acc", "PSD aceleración", False),
                   ("psd_gyr", "PSD giroscopio", False)]),
    ("Opciones", [("demean", "restar la media (ver ruido)", False)]),
]

GPS_GROUPS = [
    ("Paneles", [("track", "Traza E–N", True), ("alt", "Altitud", True),
                 ("radial", "Distancia radial", False),
                 ("speed", "Velocidad (derivada)", False)]),
    ("Datos crudos", [("lat", "Latitud", False), ("lon", "Longitud", False),
                      ("status", "Estado del fix", False),
                      ("sigma", "σ de la covarianza", False)]),
]


def fig_imu_channels(c, b: Dict, t0: float, keys: List[str]):
    from scipy.signal import welch
    n = min(c.t.size, c.accel.shape[0])
    if n < 3:
        return None
    t = c.t[:n] - t0
    a, g = c.accel[:n], np.degrees(c.gyro[:n])
    dm = "demean" in keys
    mean_lab = " − media" if dm else ""

    def prep(y):
        return y - np.nanmean(y) if dm else y

    panels: List[Tuple[str, str, list]] = []
    s = [(k, prep(a[:, i]), A.AXIS_COLOR[k]) for i, k in enumerate("xyz")
         if f"acc_{k}" in keys]
    if "acc_norm" in keys:
        s.append(("|a|", prep(np.linalg.norm(a, axis=1)), A.INK_SOFT))
    if s:
        panels.append(("Aceleración lineal", "m/s²" + mean_lab, s))
    s = [(k, prep(g[:, i]), A.AXIS_COLOR[k]) for i, k in enumerate("xyz")
         if f"gyr_{k}" in keys]
    if "gyr_norm" in keys:
        s.append(("|ω|", prep(np.linalg.norm(g, axis=1)), A.INK_SOFT))
    if s:
        panels.append(("Velocidad angular", "°/s" + mean_lab, s))
    if any(k.startswith("rpy_") for k in keys) and np.any(np.abs(c.quat[:n])):
        r = np.degrees(A.rpy_from_quat(c.quat[:n]))
        r[:, 2] = np.degrees(np.unwrap(np.radians(r[:, 2])))
        s = [(lab, prep(r[:, i]), col) for i, (lab, col) in enumerate(
            (("roll", A.BLUE), ("pitch", A.ORANGE), ("yaw", A.AQUA)))
            if f"rpy_{lab}" in keys]
        if s:
            panels.append(("Orientación", "grados" + mean_lab, s))

    fs = b["hz"] if math.isfinite(b["hz"]) and b["hz"] > 0 else 100.0
    psd = []
    if n >= 64:
        for key, arr, spec, ttl, unit, pref in (
                ("psd_acc", a, A.SPEC_ACCEL_DENSITY, "Espectro de aceleración",
                 "m/s² / √Hz", "acc"),
                ("psd_gyr", g, A.SPEC_GYRO_DENSITY, "Espectro de velocidad angular",
                 "°/s / √Hz", "gyr")):
            if key in keys:
                axes = [k for k in "xyz" if f"{pref}_{k}" in keys] or list("xyz")
                psd.append((arr, spec, ttl, unit, axes))

    rows = len(panels) + (1 if psd else 0)
    if rows == 0:
        return None
    fig = plt.figure(figsize=(13.5, max(3.2, 2.6 * len(panels) + (3.4 if psd else 0))),
                     layout="constrained")
    ratios = [1.0] * len(panels) + ([1.3] if psd else [])
    gs = fig.add_gridspec(rows, 2, height_ratios=ratios)
    first = None
    for i, (title, ylab, series) in enumerate(panels):
        ax = fig.add_subplot(gs[i, :], sharex=first)
        if first is None:
            first = ax
        for lab, y, col in series:
            A.line(ax, t, y, col, lab)
        A.finish(ax, ylab, title)
        A.stats_text(ax, "σ  " + "  ".join(f"{lab} {np.nanstd(y):.4g}"
                                            for lab, y, _ in series))
    if first is not None:
        fig.axes[len(panels) - 1].set_xlabel("tiempo [s]")
    for j, (arr, spec, ttl, unit, axes) in enumerate(psd):
        ax = fig.add_subplot(gs[len(panels), :] if len(psd) == 1
                             else gs[len(panels), j])
        nper = int(min(n, max(256, fs * 8)))
        for k in axes:
            i = "xyz".index(k)
            f, p = welch(arr[:, i] - arr[:, i].mean(), fs=fs, nperseg=nper)
            ax.loglog(f[1:], np.sqrt(p[1:]), color=A.AXIS_COLOR[k], lw=1.1, label=k)
        ax.axhline(spec, color=A.INK_MUTED, lw=1.0, ls="--", label="spec")
        ax.set_xlabel("frecuencia [Hz]")
        ax.grid(True, which="both", alpha=0.9)
        A.finish(ax, unit, ttl)
    fig.suptitle(f"{c.topic} — canales elegidos   ({n} muestras, {b['hz']:.1f} Hz)",
                 x=0.01, ha="left")
    return fig


def fig_gps_channels(c, b: Dict, t0: float, keys: List[str]):
    n = min(c.t.size, c.lat.size)
    ok = (np.isfinite(c.lat[:n]) & np.isfinite(c.lon[:n])
          & (np.abs(c.lat[:n]) > 1e-9))
    if ok.sum() < 3:
        return None
    t = (c.t[:n] - t0)[ok]
    lat, lon, alt = c.lat[:n][ok], c.lon[:n][ok], c.alt[:n][ok]
    stat, cov = c.status[:n][ok], c.cov[:n][ok]
    north, east, _, _ = A.lla_to_ne(lat, lon)
    r = np.hypot(north, east)

    # (titulo, unidad, [(etiqueta, y, color)], eje t propio o None, escalon?)
    panels: list = []
    if "alt" in keys:
        panels.append(("Altitud", "m", [("altitud", alt, A.BLUE)], None, False))
    if "lat" in keys:
        panels.append(("Latitud", "°", [("lat", lat, A.BLUE)], None, False))
    if "lon" in keys:
        panels.append(("Longitud", "°", [("lon", lon, A.ORANGE)], None, False))
    if "radial" in keys:
        static = r.max() < 10.0
        panels.append(("Dispersión radial" if static else "Distancia al centroide",
                       "m", [("radio", r, A.ORANGE)], None, False))
    if "speed" in keys:
        dt = np.diff(t)
        d = np.hypot(np.diff(north), np.diff(east))
        sp = np.where(dt > 0, d / np.where(dt > 0, dt, 1.0), np.nan)
        panels.append(("Velocidad (derivada de las posiciones)", "m/s",
                       [("v", sp, A.AQUA)], t[1:], False))
    if "status" in keys:
        panels.append(("Estado del fix", "status",
                       [("status", stat.astype(float), A.AQUA)], None, True))
    if "sigma" in keys:
        sg = np.sqrt(np.clip(cov, 0, None))
        panels.append(("σ de posición (covarianza)", "m",
                       [("este", sg[:, 0], A.BLUE), ("norte", sg[:, 1], A.ORANGE),
                        ("arriba", sg[:, 2], A.AQUA)], None, False))

    has_track = "track" in keys
    rows = max(len(panels), 1)
    if not panels and not has_track:
        return None
    fig = plt.figure(figsize=(13.5 if has_track else 11.0, max(4.6, 2.2 * rows + 1)),
                     layout="constrained")
    ncol = 2 if has_track else 1
    gs = fig.add_gridspec(rows, ncol, width_ratios=[1, 1.3] if has_track else [1])
    col0 = 1 if has_track else 0
    if has_track:
        at = fig.add_subplot(gs[:, 0])
        sc = at.scatter(east, north, c=t, cmap=A.blue_cmap(), s=4, linewidths=0)
        at.plot(east[0], north[0], "o", ms=8, mfc="none", mec=A.INK, mew=1.4)
        at.set_aspect("equal", adjustable="datalim")
        at.set_xlabel("este [m]")
        at.set_ylabel("norte [m]")
        at.set_title("Traza relativa al centroide", loc="left", pad=6)
        cb = fig.colorbar(sc, ax=at, pad=0.02, fraction=0.045)
        cb.set_label("tiempo [s]", color=A.INK_SOFT, fontsize=8.5)
        cb.outline.set_visible(False)
    first = None
    for i, (title, yl, series, tv, step) in enumerate(panels):
        ax = fig.add_subplot(gs[i, col0], sharex=first)
        if first is None:
            first = ax
        tt = t if tv is None else tv
        for lab, y, colr in series:
            if step:
                ax.step(tt, y, where="post", color=colr, lw=1.2, label=lab)
            else:
                A.line(ax, tt, y, colr, lab)
        A.finish(ax, yl, title, legend=len(series) > 1)
        yy = series[0][1]
        yy = yy[np.isfinite(yy)]
        if yy.size:
            A.stats_text(ax, f"media {yy.mean():.6g}   σ {yy.std():.4g}")
    if first is not None:
        fig.axes[-1].set_xlabel("tiempo [s]")
    fig.suptitle(f"{c.topic} — canales elegidos   ({int(ok.sum())} fixes, "
                 f"{b['hz']:.1f} Hz)", x=0.01, ha="left")
    return fig


def align_time(c) -> None:
    """Si el tope --sample corto la deserializacion, alinea t con los datos."""
    name = {"imu": "accel", "odom": "pos", "fix": "lat"}.get(c.kind)
    if name is None:
        return
    m = len(getattr(c, name))
    if c.t.size != m:
        use_hdr = c.t_hdr.size >= m and m > 0 and np.any(c.t_hdr[:m] > 0)
        c.t = c.t_hdr[:m] if use_hdr else c.t_bag[:m]


def has_data(c) -> bool:
    k = c.kind
    if k == "imu":
        return c.accel.size > 0
    if k in ("odom", "car", "pose"):
        return c.pos.size > 0
    if k == "twist":
        return c.vel.size > 0
    if k == "fix":
        return c.lat.size > 0
    if k == "cloud":
        return c.t_bag.size > 0
    if k == "scan":
        return c.t_bag.size > 0
    if k == "image":
        return c.t_bag.size > 0
    if k == "range":
        return c.range.size > 0
    if k == "battery":
        return c.data.size > 0
    if k == "joint":
        return bool(c.names)
    if k == "diag":
        return bool(c.rows)
    if k == "clock":
        return c.clock.size > 0
    if k == "scalar":
        return c.value.size > 0
    if k in ("posearray", "path"):
        v = getattr(c, "counts", None)
        return v is not None and v.size > 0
    if k == "tf":
        return bool(c.tf)
    if k == "log":
        return bool(c.rows)
    return bool(getattr(c, "data", None))


# --------------------------------------------------------------------------- #
#  Dialogos y ventana de grafica
# --------------------------------------------------------------------------- #

class PlotWindow(tk.Toplevel):
    """Una figura de matplotlib en su propia ventana, con barra de herramientas."""

    def __init__(self, master, fig, title: str):
        super().__init__(master)
        self.title(title)
        self.fig = fig
        wi, hi = fig.get_size_inches()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(1300, sw - 80)
        h = min(int(w * hi / wi) + 50, sh - 120)
        self.geometry(f"{w}x{max(h, 380)}")
        canvas = FigureCanvasTkAgg(fig, master=self)
        bar = NavigationToolbar2Tk(canvas, self, pack_toolbar=False)
        bar.update()
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw_idle()
        self.canvas = canvas
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self) -> None:
        plt.close(self.fig)
        self.destroy()


class ListDialog(tk.Toplevel):
    def __init__(self, master, title: str, items: List[str], multi: bool = False):
        super().__init__(master)
        self.title(title)
        self.result: List[str] = []
        self.transient(master)
        self.lb = tk.Listbox(self, selectmode=tk.EXTENDED if multi else tk.SINGLE,
                             width=64, height=min(12, max(3, len(items))),
                             exportselection=False)
        for it in items:
            self.lb.insert(tk.END, it)
        self.lb.selection_set(0)
        self.lb.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.lb.bind("<Double-Button-1>", lambda _e: self._ok())
        row = ttk.Frame(self)
        row.pack(pady=(0, 10))
        ttk.Button(row, text="Aceptar", command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Cancelar", command=self.destroy).pack(side=tk.LEFT, padx=4)
        self.items = items
        self._modal()

    def _ok(self) -> None:
        self.result = [self.items[i] for i in self.lb.curselection()]
        self.destroy()

    def _modal(self) -> None:
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass
        self.wait_window(self)


class CheckDialog(tk.Toplevel):
    def __init__(self, master, title: str, groups):
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self.result: Optional[List[str]] = None
        self.vars: Dict[str, tk.BooleanVar] = {}
        for gname, items in groups:
            lf = ttk.LabelFrame(self, text=gname, padding=6)
            lf.pack(fill=tk.X, padx=10, pady=4)
            for key, label, default in items:
                v = tk.BooleanVar(value=default)
                self.vars[key] = v
                ttk.Checkbutton(lf, text=label, variable=v).pack(side=tk.LEFT, padx=6)
        row = ttk.Frame(self)
        row.pack(pady=8)
        ttk.Button(row, text="Graficar", command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Cancelar", command=self.destroy).pack(side=tk.LEFT, padx=4)
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass
        self.wait_window(self)

    def _ok(self) -> None:
        self.result = [k for k, v in self.vars.items() if v.get()]
        self.destroy()


class ReplayPanel(ttk.Frame):
    """Reproduce el bag y compara en vivo sensor solo vs EKF, dentro de la GUI.

    El canvas de matplotlib se incrusta en esta pestana y se refresca con
    `after`; no se lanza ninguna ventana ni proceso grafico aparte. Los
    procesos del nodo y de `ros2 bag play` se controlan desde aqui.
    """

    def __init__(self, master):
        super().__init__(master, padding=8)
        self.running = False
        self.dash = None
        self.canvas = None
        self.placeholder = None
        self.store = None
        self.live = None
        self.stop_evt = None
        self.spin = None
        self.node_proc = None
        self.play_proc = None
        self._poll_after = None
        self._start_after = None
        self._owns_ctx = False

        self.bag = tk.StringVar()
        self.raw = tk.StringVar(value=R.DEFAULT_RAW if R else "/sensors/imu/odom")
        self.filt = tk.StringVar(value=R.DEFAULT_FILTERED if R else "/state/car")
        self.node_cmd = tk.StringVar(
            value=R.DEFAULT_NODE if R else "ros2 run mirena_state mirena_state")
        self.rate = tk.StringVar(value="1.0")
        self.delay = tk.StringVar(value="2.0")
        self.start_s = tk.StringVar()
        self.duration = tk.StringVar()
        self.loop = tk.BooleanVar(value=False)
        self.no_node = tk.BooleanVar(value=False)
        self.no_play = tk.BooleanVar(value=False)
        self.no_align = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Elige un bag y pulsa Iniciar.")

        self._build()

    # ---- construccion ----------------------------------------------------- #

    def _build(self) -> None:
        cfg = ttk.LabelFrame(self, text="Bag y nodo", padding=8)
        cfg.pack(fill=tk.X)

        ttk.Label(cfg, text="Bag").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(cfg, textvariable=self.bag, width=14).grid(
            row=0, column=1, columnspan=5, sticky="we", padx=6, pady=2)
        btns = ttk.Frame(cfg)
        btns.grid(row=0, column=6, columnspan=2, sticky="e", pady=2)
        ttk.Button(btns, text="Carpeta…", width=8,
                   command=self._ask_folder).pack(side=tk.LEFT)
        ttk.Button(btns, text="Fichero…", width=8,
                   command=self._ask_file).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(cfg, text="Sensor solo").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(cfg, textvariable=self.raw, width=18).grid(
            row=1, column=1, sticky="we", padx=6)
        ttk.Label(cfg, text="Estado EKF").grid(row=1, column=2, sticky="w", padx=(8, 0))
        ttk.Entry(cfg, textvariable=self.filt, width=14).grid(
            row=1, column=3, sticky="we", padx=6)

        ttk.Label(cfg, text="Nodo").grid(row=2, column=0, sticky="w", pady=2)
        self.node_entry = ttk.Entry(cfg, textvariable=self.node_cmd)
        self.node_entry.grid(row=2, column=1, columnspan=5, sticky="we", padx=6)

        ttk.Label(cfg, text="Velocidad").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(cfg, textvariable=self.rate, width=7).grid(
            row=3, column=1, sticky="w", padx=6)
        ttk.Label(cfg, text="Espera [s]").grid(row=3, column=2, sticky="w", padx=(8, 0))
        ttk.Entry(cfg, textvariable=self.delay, width=7).grid(
            row=3, column=3, sticky="w", padx=6)
        ttk.Label(cfg, text="Inicio [s]").grid(row=3, column=4, sticky="w", padx=(8, 0))
        ttk.Entry(cfg, textvariable=self.start_s, width=7).grid(
            row=3, column=5, sticky="w", padx=6)
        ttk.Label(cfg, text="Duración [s]").grid(row=3, column=6, sticky="w", padx=(8, 0))
        ttk.Entry(cfg, textvariable=self.duration, width=7).grid(
            row=3, column=7, sticky="w", padx=6)

        opts = ttk.Frame(cfg)
        opts.grid(row=4, column=0, columnspan=8, sticky="w", pady=(6, 0))
        ttk.Checkbutton(opts, text="No arrancar nodo", variable=self.no_node,
                        command=self._toggle_node).pack(side=tk.LEFT)
        ttk.Checkbutton(opts, text="No reproducir",
                        variable=self.no_play).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(opts, text="Bucle", variable=self.loop).pack(
            side=tk.LEFT, padx=10)
        ttk.Checkbutton(opts, text="No alinear", variable=self.no_align).pack(
            side=tk.LEFT, padx=10)

        cfg.columnconfigure(1, weight=3)
        cfg.columnconfigure(3, weight=2)
        cfg.columnconfigure(5, weight=1)
        cfg.columnconfigure(7, weight=1)

        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=6)
        self.btn_start = ttk.Button(row, text="Iniciar", command=self.start)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(row, text="Detener", command=self.stop,
                                   state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=6)
        self.btn_save = ttk.Button(row, text="Guardar PNG…", command=self.save_png,
                                   state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.status,
                  foreground=A.INK_SOFT).pack(side=tk.LEFT, padx=12)

        self.host = ttk.Frame(self, relief="sunken", borderwidth=1)
        self.host.pack(fill=tk.BOTH, expand=True)
        self._placeholder()

    def _placeholder(self) -> None:
        if self.placeholder is None:
            self.placeholder = ttk.Label(
                self.host, anchor="center",
                text="La gráfica en vivo aparecerá aquí.\n"
                     "Configura el bag y pulsa Iniciar.")
            self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def _toggle_node(self) -> None:
        self.node_entry.configure(
            state=tk.DISABLED if self.no_node.get() else tk.NORMAL)

    def _ask_folder(self) -> None:
        p = filedialog.askdirectory(title="Carpeta del bag (con metadata.yaml)")
        if p:
            self.bag.set(p)

    def _ask_file(self) -> None:
        p = filedialog.askopenfilename(
            title="Fichero de bag",
            filetypes=[("rosbag2", "*.mcap *.db3"), ("todos", "*.*")])
        if p:
            self.bag.set(p)

    # ---- configuracion externa ------------------------------------------- #

    def set_bag(self, path: str) -> None:
        if not path:
            return
        if not self.running:
            self.bag.set(path)

    def configure_bag(self, path: str, raw: str, filt: str) -> None:
        self.set_bag(path)
        if raw:
            self.raw.set(raw)
        if filt:
            self.filt.set(filt)

    # ---- ciclo de vida ---------------------------------------------------- #

    @staticmethod
    def _num(var: tk.StringVar, name: str, default=None):
        s = var.get().strip()
        if not s:
            return default
        try:
            return float(s)
        except ValueError as exc:
            raise ValueError(f"«{name}» debe ser numérico.") from exc

    def _clear_plot(self) -> None:
        if self.canvas is not None:
            try:
                self.canvas.get_tk_widget().destroy()
            except Exception:  # noqa: BLE001
                pass
            self.canvas = None
        if self.dash is not None:
            try:
                plt.close(self.dash.fig)
            except Exception:  # noqa: BLE001
                pass
            self.dash = None
        self.placeholder = None
        self._placeholder()

    def _embed(self) -> None:
        if self.placeholder is not None:
            self.placeholder.destroy()
            self.placeholder = None
        self.canvas = FigureCanvasTkAgg(self.dash.fig, master=self.host)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def start(self) -> None:
        if self.running:
            return
        if R is None:
            messagebox.showerror("Falta bag_replay.py",
                                 "No encuentro bag_replay.py junto a rosbag_gui.py.")
            return
        bag = os.path.expanduser(self.bag.get().strip())
        if not bag or not os.path.exists(bag):
            messagebox.showerror(
                "Bag no válido",
                "Elige la carpeta del bag o un fichero .mcap existente.")
            return
        if os.path.isfile(bag) and not bag.lower().endswith((".mcap", ".db3")):
            messagebox.showerror(
                "Bag no válido",
                "El fichero debe ser .mcap (o .db3). Para una carpeta usa "
                "«Carpeta…».")
            return
        try:
            rate = self._num(self.rate, "velocidad", 1.0)
            delay = self._num(self.delay, "espera", 0.0) or 0.0
            start = self._num(self.start_s, "inicio")
            duration = self._num(self.duration, "duración")
        except ValueError as exc:
            messagebox.showerror("Valor no válido", str(exc))
            return
        if rate <= 0:
            messagebox.showerror("Valor no válido", "La velocidad debe ser > 0.")
            return
        if delay < 0:
            messagebox.showerror("Valor no válido", "La espera no puede ser negativa.")
            return

        try:
            import rclpy
        except ImportError:
            messagebox.showerror(
                "Falta rclpy",
                "Sourcea ROS 2 antes de abrir la GUI:\n\n"
                "    source /opt/ros/jazzy/setup.bash\n"
                "    source ~/mirena_ws/install/setup.bash")
            return

        self.status.set("Arrancando nodo y suscripciones…")
        self.update_idletasks()
        try:
            self._clear_plot()
            self.store = R.Store()
            self._owns_ctx = False
            try:
                rclpy.init()
                self._owns_ctx = True
            except Exception:  # ya inicializado por otro componente
                pass
            raw_topic = self.raw.get().strip()
            filt_topic = self.filt.get().strip()
            self.live = R.LiveNode(self.store, raw_topic, filt_topic)
            self.dash = R.Dashboard(raw_topic, filt_topic,
                                    align=not self.no_align.get())
            self._embed()
            self.stop_evt = threading.Event()
            self.spin = threading.Thread(target=self.live.spin,
                                         args=(self.stop_evt,), daemon=True)
            self.spin.start()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._teardown()
            self._clear_plot()
            self.status.set("Error al iniciar.")
            messagebox.showerror("No pude iniciar la reproducción",
                                 f"{type(exc).__name__}: {exc}")
            return

        self.running = True
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_save.configure(state=tk.NORMAL)

        if not self.no_node.get():
            try:
                cmd = shlex.split(self.node_cmd.get().strip())
            except ValueError as exc:
                messagebox.showerror("Comando no válido", str(exc))
                self.stop()
                return
            if not cmd:
                messagebox.showerror("Comando no válido",
                                     "Escribe el comando del nodo o marca "
                                     "«No arrancar nodo».")
                self.stop()
                return
            self.node_proc = R.spawn(cmd, "nodo")
            if self.node_proc is None:
                self.stop()
                self.status.set("No pude arrancar el nodo.")
                return

        play_cmd = None
        if not self.no_play.get():
            bag_uri, bag_storage = R.bag_target(bag)
            play_cmd = ["ros2", "bag", "play", bag_uri, "--rate", f"{rate:g}"]
            if bag_storage:
                play_cmd += ["--storage", bag_storage]
            if start is not None:
                play_cmd += ["--start-offset", f"{start:g}"]
            if duration is not None:
                play_cmd += ["--playback-duration", f"{duration:g}"]
            if self.loop.get():
                play_cmd += ["--loop"]

        def launch() -> None:
            self._start_after = None
            if not self.running:
                return
            if play_cmd is not None:
                self.play_proc = R.spawn(play_cmd, "bag")
                if self.play_proc is None:
                    messagebox.showerror(
                        "No pude reproducir",
                        "No encontré `ros2 bag play`.\n\n"
                        "Abre la GUI después de sourcear ROS 2 o marca "
                        "«No reproducir».")
                    self.stop()
                    return
            self.status.set("Reproduciendo: sensor solo vs EKF…")

        if delay > 0 and not self.no_node.get():
            self.status.set(f"Esperando {delay:g} s a que arranque el nodo…")
            self._start_after = self.after(int(delay * 1000), launch)
        else:
            launch()
        self._poll_after = self.after(120, self._poll)

    def _teardown(self) -> None:
        if self.stop_evt is not None:
            self.stop_evt.set()
        if R is not None:
            R.stop_proc(self.play_proc, "bag")
            R.stop_proc(self.node_proc, "nodo")
        self.play_proc = None
        self.node_proc = None
        if self.spin is not None:
            self.spin.join(timeout=2.5)
            self.spin = None
        self.stop_evt = None
        if self._owns_ctx:
            try:
                import rclpy
                rclpy.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._owns_ctx = False
        self.store = None
        self.live = None

    def stop(self, auto: bool = False) -> None:
        if self._start_after is not None:
            try:
                self.after_cancel(self._start_after)
            except Exception:  # noqa: BLE001
                pass
            self._start_after = None
        if self._poll_after is not None:
            try:
                self.after_cancel(self._poll_after)
            except Exception:  # noqa: BLE001
                pass
            self._poll_after = None
        if not self.running:
            return
        raw = filt = None
        if self.store is not None:
            raw, filt = self.store.snapshot()[:2]
        self.running = False
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_save.configure(
            state=tk.NORMAL if self.dash is not None else tk.DISABLED)
        self._teardown()
        nr = raw["t"].size if raw is not None else 0
        nf = filt["t"].size if filt is not None else 0
        prefix = "Reproducción terminada" if auto else "Detenido"
        self.status.set(f"{prefix} · sensor {nr} · EKF {nf} muestras")

    def _poll(self) -> None:
        self._poll_after = None
        if not self.running or self.store is None or self.dash is None:
            return
        raw, filt, t0, idle = self.store.snapshot()
        try:
            self.dash.update(raw, filt, t0, idle)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        nr, nf = raw["t"].size, filt["t"].size
        if idle >= 2.0:
            estado = "en espera"
        elif self.play_proc is not None:
            estado = "reproduciendo"
        else:
            estado = "escuchando"
        self.status.set(f"{estado} · sensor {nr} · EKF {nf} muestras")
        if (self.play_proc is not None and self.play_proc.poll() is not None
                and idle > 2.0 and not self.loop.get()):
            self.stop(auto=True)
            return
        self._poll_after = self.after(120, self._poll)

    def save_png(self) -> None:
        if self.dash is None:
            messagebox.showinfo("Sin gráfica", "Todavía no hay datos que guardar.")
            return
        p = filedialog.asksaveasfilename(
            title="Guardar dashboard", defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if p:
            R.save_dashboard(self.dash, p)
            self.status.set(f"Guardado {p}")

    def shutdown(self) -> None:
        self.stop()
        self._clear_plot()


def make_text(parent) -> tk.Text:
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.BOTH, expand=True)
    txt = tk.Text(frame, wrap="none", font="TkFixedFont", state="disabled")
    sy = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
    sx = ttk.Scrollbar(frame, orient="horizontal", command=txt.xview)
    txt.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
    sy.pack(side=tk.RIGHT, fill=tk.Y)
    sx.pack(side=tk.BOTTOM, fill=tk.X)
    txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    return txt


def set_text(txt: tk.Text, s: str) -> None:
    txt.configure(state="normal")
    txt.delete("1.0", tk.END)
    txt.insert("1.0", s)
    txt.configure(state="disabled")


# --------------------------------------------------------------------------- #
#  Aplicacion
# --------------------------------------------------------------------------- #

class App(tk.Tk):
    def __init__(self, path: Optional[str] = None):
        super().__init__()
        self.title("ROS 2 bag analyzer")
        self.geometry("1320x800")
        self.col: Dict[str, A.Base] = {}
        self.qos: Dict[str, str] = {}
        self.st: Dict[str, Dict] = {}
        self.t0 = 0.0
        self.path = ""
        self.q: "queue.Queue" = queue.Queue()
        self.loading = False
        self._poll_id: Optional[str] = None

        self.only_nav = tk.BooleanVar(value=False)
        self.sample = tk.StringVar(value="1000000")
        self.t_start = tk.StringVar()
        self.t_end = tk.StringVar()
        self.status = tk.StringVar(value="Abre un bag para empezar.")

        self._menu()
        self._ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_id = self.after(100, self._poll)
        if path:
            self.after(200, lambda: self.open_bag(path))

    # ---- construccion ----------------------------------------------------- #

    def _menu(self) -> None:
        m = tk.Menu(self)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Abrir carpeta de bag…", command=self.ask_folder)
        f.add_command(label="Abrir fichero .mcap / .db3…", command=self.ask_file)
        f.add_separator()
        f.add_command(label="Exportar CSV de la selección…", command=self.export_csv)
        f.add_separator()
        f.add_command(label="Salir", command=self._on_close)
        m.add_cascade(label="Archivo", menu=f)
        self.config(menu=m)

    def _ui(self) -> None:
        top = ttk.Frame(self, padding=6)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Abrir bag…", command=self.ask_folder).pack(side=tk.LEFT)
        ttk.Button(top, text="Abrir fichero…", command=self.ask_file).pack(
            side=tk.LEFT, padx=(4, 12))
        ttk.Checkbutton(top, text="Solo IMU + GPS (carga rápida)",
                        variable=self.only_nav).pack(side=tk.LEFT)
        ttk.Label(top, text="   máx. msgs/topic").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.sample, width=9).pack(side=tk.LEFT, padx=4)
        self.prog = ttk.Progressbar(top, mode="indeterminate", length=160)
        self.prog.pack(side=tk.RIGHT)

        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # --- izquierda: topics + acciones
        left = ttk.Frame(pw)
        pw.add(left, weight=2)
        cols = ("type", "msgs", "hz", "gaps")
        self.tree = ttk.Treeview(left, columns=cols, show="tree headings",
                                 selectmode="extended", height=12)
        self.tree.heading("#0", text="topic")
        self.tree.column("#0", width=230)
        for k, txt, w in (("type", "tipo", 110), ("msgs", "msgs", 60),
                          ("hz", "Hz", 55), ("gaps", "huecos", 55)):
            self.tree.heading(k, text=txt)
            self.tree.column(k, width=w, anchor="e" if k != "type" else "w")
        ys = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        ys.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.on_select())
        self.tree.bind("<Double-Button-1>", lambda _e: self.plot_selected())

        acts = ttk.Frame(self)
        acts.pack(in_=left, fill=tk.X, pady=(6, 0))

        imu = ttk.LabelFrame(acts, text="Solo IMU", padding=6)
        imu.pack(fill=tk.X, pady=2)
        ttk.Button(imu, text="Dashboard", command=self.imu_dashboard).pack(side=tk.LEFT, padx=2)
        ttk.Button(imu, text="Canales…", command=self.imu_channels).pack(side=tk.LEFT, padx=2)
        ttk.Button(imu, text="Gravedad 3D", command=self.imu_3d).pack(side=tk.LEFT, padx=2)

        gps = ttk.LabelFrame(acts, text="Solo GPS", padding=6)
        gps.pack(fill=tk.X, pady=2)
        ttk.Button(gps, text="Dashboard", command=self.gps_dashboard).pack(side=tk.LEFT, padx=2)
        ttk.Button(gps, text="Canales…", command=self.gps_channels).pack(side=tk.LEFT, padx=2)
        ttk.Button(gps, text="Traza 3D", command=self.gps_3d).pack(side=tk.LEFT, padx=2)

        sel = ttk.LabelFrame(acts, text="Topic seleccionado", padding=6)
        sel.pack(fill=tk.X, pady=2)
        ttk.Button(sel, text="Graficar", command=self.plot_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(sel, text="3D", command=self.plot_3d_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(sel, text="Temporización", command=self.plot_timing).pack(side=tk.LEFT, padx=2)
        ttk.Button(sel, text="CSV…", command=self.export_csv).pack(side=tk.LEFT, padx=2)

        glob = ttk.LabelFrame(acts, text="Global", padding=6)
        glob.pack(fill=tk.X, pady=2)
        ttk.Button(glob, text="Tasas y ancho de banda",
                   command=self.plot_global).pack(side=tk.LEFT, padx=2)
        ttk.Button(glob, text="Comparar IMU",
                   command=self.compare_imus).pack(side=tk.LEFT, padx=2)
        ttk.Button(glob, text="Reproducir + Kalman",
                   command=self.replay).pack(side=tk.LEFT, padx=2)

        win = ttk.LabelFrame(acts, text="Ventana temporal [s desde el inicio]", padding=6)
        win.pack(fill=tk.X, pady=2)
        ttk.Label(win, text="inicio").pack(side=tk.LEFT)
        ttk.Entry(win, textvariable=self.t_start, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(win, text="fin").pack(side=tk.LEFT)
        ttk.Entry(win, textvariable=self.t_end, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(win, text="(vacío = todo)").pack(side=tk.LEFT)

        # --- derecha: informes
        right = ttk.Notebook(pw)
        pw.add(right, weight=3)
        f1, f2 = ttk.Frame(right), ttk.Frame(right)
        right.add(f1, text="Informe del topic")
        right.add(f2, text="Resumen general")
        self.txt_topic = make_text(f1)
        self.txt_over = make_text(f2)
        self.replay_panel = ReplayPanel(right)
        right.add(self.replay_panel, text="Reproducción + Kalman")
        self.nb = right

        ttk.Label(self, textvariable=self.status, relief="sunken",
                  anchor="w", padding=(6, 2)).pack(fill=tk.X, side=tk.BOTTOM)

    # ---- carga ------------------------------------------------------------ #

    def ask_folder(self) -> None:
        p = filedialog.askdirectory(title="Carpeta del bag (con metadata.yaml)")
        if p:
            self.open_bag(p)

    def ask_file(self) -> None:
        p = filedialog.askopenfilename(
            title="Fichero de bag", filetypes=[("rosbag2", "*.mcap *.db3"), ("todos", "*.*")])
        if p:
            self.open_bag(p)

    def open_bag(self, path: str) -> None:
        if self.loading:
            return
        try:
            sample = max(1000, int(self.sample.get()))
        except ValueError:
            messagebox.showerror("Valor no válido", "«máx. msgs/topic» debe ser un entero.")
            return
        self.loading = True
        self.path = path
        self.replay_panel.set_bag(path)
        self.status.set(f"Leyendo {path} …")
        self.prog.start(12)
        threading.Thread(target=self._load_worker,
                         args=(path, self.only_nav.get(), sample), daemon=True).start()

    def _load_worker(self, path: str, only_nav: bool, sample: int) -> None:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                wanted = None
                if only_nav:
                    r = A.open_reader(path)
                    wanted = [t.name for t in r.get_all_topics_and_types()
                              if t.type in (A.IMU_TYPE, A.FIX_TYPE)]
                    del r
                    if not wanted:
                        raise RuntimeError("El bag no tiene topics Imu ni NavSatFix.")
                col, qos = A.scan(path, wanted, sample)
                if not col:
                    raise RuntimeError("El bag no tiene topics legibles.")
                for c in col.values():
                    align_time(c)
                t0 = min((c.t_bag[0] for c in col.values() if c.t_bag.size), default=0.0)
                st = A.report_overview(col, qos)
                for c in col.values():
                    if c.kind == "log":
                        A.report_log(c, 60)
            self.q.put(("ok", (col, qos, st, t0, buf.getvalue())))
        except SystemExit as exc:
            self.q.put(("err", str(exc).strip() or "No se pudo abrir el bag."))
        except Exception:  # noqa: BLE001
            self.q.put(("err", traceback.format_exc()))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                self.prog.stop()
                self.loading = False
                if kind == "err":
                    self.status.set("Error al cargar.")
                    messagebox.showerror("No se pudo cargar el bag", payload)
                    continue
                self.col, self.qos, self.st, self.t0, over = payload
                set_text(self.txt_over, over)
                set_text(self.txt_topic, "Selecciona un topic de la lista.")
                self.tree.delete(*self.tree.get_children())
                for n in sorted(self.col):
                    c, b = self.col[n], self.st[n]
                    hz = f"{b['hz']:.1f}" if math.isfinite(b["hz"]) else "-"
                    self.tree.insert("", "end", iid=n, text=n,
                                     values=(c.type.split("/")[-1], b["n"], hz, b["gaps"]))
                imus = sum(c.kind == "imu" for c in self.col.values())
                gps = sum(c.kind == "fix" for c in self.col.values())
                raw = next((n for n, c in self.col.items() if c.kind == "odom"), None)
                car = next((n for n, c in self.col.items() if c.kind == "car"), None)
                self.replay_panel.configure_bag(
                    self.path, raw or self.replay_panel.raw.get(),
                    car or self.replay_panel.filt.get())
                self.status.set(f"{self.path}   ·   {len(self.col)} topics   ·   "
                                f"{imus} IMU, {gps} GPS")
        except queue.Empty:
            pass
        self._poll_id = self.after(100, self._poll)

    # ---- utilidades ------------------------------------------------------- #

    def _need_bag(self) -> bool:
        if not self.col:
            messagebox.showinfo("Sin datos", "Abre un bag primero.")
            return False
        return True

    def _window(self) -> Tuple[Optional[float], Optional[float]]:
        def val(s: str) -> Optional[float]:
            s = s.strip()
            return float(s) if s else None
        try:
            return val(self.t_start.get()), val(self.t_end.get())
        except ValueError:
            raise ValueError("La ventana temporal debe ser numérica (segundos).")

    def view(self, name: str):
        """Colector y estadisticas, recortados a la ventana temporal si hay."""
        c, b = self.col[name], self.st[name]
        s, e = self._window()
        if s is None and e is None:
            return c, b
        c2 = copy.copy(c)
        A.window(c2, self.t0, s, e)
        if c2.t.size < 2:
            raise ValueError(f"No hay datos de {name} en esa ventana temporal.")
        return c2, A.base_stats(c2)

    def pick(self, kind: str, multi: bool = False) -> List[str]:
        if not self._need_bag():
            return []
        cands = [n for n, c in self.col.items() if c.kind == kind and has_data(c)]
        label = {"imu": "IMU", "fix": "GPS"}.get(kind, kind)
        if not cands:
            hint = "" if not self.only_nav.get() else ""
            messagebox.showinfo("Sin topics", f"Este bag no tiene topics {label} con datos.{hint}")
            return []
        sel = [n for n in self.tree.selection() if n in cands]
        if sel:
            return sel if multi else sel[:1]
        if len(cands) == 1:
            return cands
        return ListDialog(self, f"Elige topic {label}", cands, multi).result

    def show(self, builder, title: str) -> None:
        try:
            fig = builder()
            if fig is None:
                messagebox.showinfo("Sin gráfica", "No hay datos suficientes para esa gráfica.")
                return
            PlotWindow(self, fig, title)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror("Error al graficar", f"{type(exc).__name__}: {exc}")

    def captured(self, fn, *a, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = fn(*a, **kw)
        return r, buf.getvalue()

    # ---- informe ---------------------------------------------------------- #

    def on_select(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        n = sel[0]
        try:
            c, b = self.col[n], self.st[n]
        except KeyError:
            return
        _, text = self.captured(self._report, c, b)
        set_text(self.txt_topic, text)
        self.nb.select(0)

    @staticmethod
    def _report(c, b) -> None:
        if not has_data(c) and c.kind != "cloud":
            print(f"{c.topic}: {b['n']} mensajes, {b['hz']:.2f} Hz. "
                  f"Sin campos deserializables.")
        elif c.kind == "imu":
            A.report_imu(c, b, A.imu_stats(c))
        elif c.kind == "odom":
            A.report_odom(c, b)
        elif c.kind == "car":
            A.report_car(c, b)
        elif c.kind == "pose":
            A.report_pose(c, b)
        elif c.kind == "twist":
            A.report_twist(c, b)
        elif c.kind == "fix":
            A.report_fix(c, b)
        elif c.kind == "cloud":
            A.report_cloud(c, b)
        elif c.kind == "scan":
            A.report_scan(c, b)
        elif c.kind == "image":
            A.report_image(c, b)
        elif c.kind == "range":
            A.report_range(c, b)
        elif c.kind == "battery":
            A.report_battery(c, b)
        elif c.kind == "joint":
            A.report_joint(c, b)
        elif c.kind == "diag":
            A.report_diag(c, b)
        elif c.kind == "clock":
            A.report_clock(c, b)
        elif c.kind == "scalar":
            A.report_scalar(c, b)
        elif c.kind == "posearray":
            A.report_posearray(c, b)
        elif c.kind == "path":
            A.report_path(c, b)
        elif c.kind == "tf":
            A.report_tf(c)
        elif c.kind == "log":
            A.report_log(c, 60)
        else:
            A.report_generic(c, b)

    # ---- IMU -------------------------------------------------------------- #

    def imu_dashboard(self) -> None:
        for n in self.pick("imu", multi=True):
            def build(n=n):
                c, b = self.view(n)
                return A.fig_imu(c, b, A.imu_stats(c), self.t0)
            self.show(build, f"IMU — {n}")

    def imu_channels(self) -> None:
        for n in self.pick("imu"):
            dlg = CheckDialog(self, f"Canales de {n}", IMU_GROUPS)
            if not dlg.result:
                return
            keys = dlg.result

            def build(n=n, keys=keys):
                c, b = self.view(n)
                return fig_imu_channels(c, b, self.t0, keys)
            self.show(build, f"IMU canales — {n}")

    def imu_3d(self) -> None:
        names = self.pick("imu", multi=True)
        if not names:
            return

        def build():
            imus, stats = {}, {}
            for n in names:
                c, _ = self.view(n)
                imus[n], stats[n] = c, A.imu_stats(c)
            return A.fig_imu3d(imus, stats)
        self.show(build, "IMU — gravedad 3D")

    def compare_imus(self) -> None:
        if not self._need_bag():
            return
        cands = [n for n, c in self.col.items() if c.kind == "imu" and has_data(c)]
        if len(cands) < 2:
            messagebox.showinfo("Comparar IMU", "Hacen falta al menos dos topics IMU.")
            return
        sel = [n for n in self.tree.selection() if n in cands]
        names = sel if len(sel) >= 2 else ListDialog(
            self, "Elige dos o más IMU", cands, multi=True).result
        if len(names) < 2:
            return
        try:
            imus, stats = {}, {}
            for n in names:
                c, _ = self.view(n)
                imus[n], stats[n] = c, A.imu_stats(c)
            rot: Dict = {}
            notes, text = self.captured(A.report_imu_compare, imus, stats, rot)
            set_text(self.txt_topic, text)
            self.nb.select(0)
            PlotWindow(self, A.fig_imu_compare(imus, self.t0, notes), "Comparación de IMU")
            PlotWindow(self, A.fig_imu3d(imus, stats, rot or None), "IMU — 3D")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror("Error", f"{type(exc).__name__}: {exc}")

    # ---- reproduccion en vivo --------------------------------------------- #

    def replay(self) -> None:
        if not self._need_bag():
            return
        bag = self.path
        if not bag:
            return
        raw = next((n for n, c in self.col.items() if c.kind == "odom"),
                   "/sensors/imu/odom")
        filt = next((n for n, c in self.col.items() if c.kind == "car"),
                    "/state/car")
        self.replay_panel.configure_bag(bag, raw, filt)
        self.nb.select(self.replay_panel)
        if R is None:
            messagebox.showerror("Falta bag_replay.py",
                                 "No encuentro bag_replay.py junto a rosbag_gui.py.")
            return
        self.status.set("Reproducción integrada lista: pulsa Iniciar en la pestaña.")

    def _on_close(self) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except Exception:  # noqa: BLE001
                pass
            self._poll_id = None
        try:
            self.replay_panel.shutdown()
        finally:
            self.destroy()

    # ---- GPS -------------------------------------------------------------- #

    def gps_dashboard(self) -> None:
        for n in self.pick("fix", multi=True):
            def build(n=n):
                c, b = self.view(n)
                return A.fig_fix(c, b, self.t0)
            self.show(build, f"GPS — {n}")

    def gps_channels(self) -> None:
        for n in self.pick("fix"):
            dlg = CheckDialog(self, f"Canales de {n}", GPS_GROUPS)
            if not dlg.result:
                return
            keys = dlg.result

            def build(n=n, keys=keys):
                c, b = self.view(n)
                return fig_gps_channels(c, b, self.t0, keys)
            self.show(build, f"GPS canales — {n}")

    def gps_3d(self) -> None:
        for n in self.pick("fix", multi=True):
            def build(n=n):
                c, b = self.view(n)
                return A.fig_fix3d(c, b, self.t0)
            self.show(build, f"GPS 3D — {n}")

    # ---- topic seleccionado / global ------------------------------------- #

    def _selected(self) -> List[str]:
        if not self._need_bag():
            return []
        sel = list(self.tree.selection())
        if not sel:
            messagebox.showinfo("Selecciona un topic", "Elige uno o más topics de la lista.")
        return sel

    def plot_selected(self) -> None:
        for n in self._selected():
            def build(n=n):
                c, b = self.view(n)
                if c.kind == "imu" and has_data(c):
                    return A.fig_imu(c, b, A.imu_stats(c), self.t0)
                if c.kind == "odom" and has_data(c):
                    cons, _ = self.captured(A.report_odom, c, b)
                    return A.fig_odom(c, b, self.t0, cons)
                if c.kind == "car" and has_data(c):
                    return A.fig_car(c, b, self.t0)
                if c.kind == "pose" and has_data(c):
                    return A.fig_pose(c, b, self.t0)
                if c.kind == "twist" and has_data(c):
                    return A.fig_twist(c, b, self.t0)
                if c.kind == "fix" and has_data(c):
                    return A.fig_fix(c, b, self.t0)
                if c.kind == "cloud":
                    return A.fig_cloud(c, b, self.t0)
                if c.kind == "scan":
                    return A.fig_scan(c, b, self.t0)
                if c.kind == "image":
                    return A.fig_image(c, b, self.t0)
                if c.kind == "range":
                    return A.fig_range(c, b, self.t0)
                if c.kind == "battery":
                    return A.fig_battery(c, b, self.t0)
                if c.kind == "joint":
                    return A.fig_joint(c, b, self.t0)
                if c.kind == "diag":
                    return A.fig_diag(c, b, self.t0)
                if c.kind == "clock":
                    return A.fig_clock(c, b, self.t0)
                if c.kind == "scalar":
                    return A.fig_scalar(c, b, self.t0)
                if c.kind == "posearray":
                    return A.fig_posearray(c, b, self.t0)
                if c.kind == "path":
                    return A.fig_path(c, b, self.t0)
                if c.kind == "tf":
                    return A.fig_tf3d(c)
                if c.kind == "log":
                    raise ValueError("/rosout no tiene gráfica; mira el resumen general.")
                if getattr(c, "data", None):
                    return A.fig_generic(c, b, self.t0)
                return None
            self.show(build, n)

    def plot_3d_selected(self) -> None:
        for n in self._selected():
            def build(n=n):
                c, b = self.view(n)
                if c.kind == "odom" and has_data(c):
                    return A.fig_traj3d(c, b, self.t0)
                if c.kind in ("car", "pose") and has_data(c):
                    return A.fig_traj3d(c, b, self.t0)
                if c.kind == "fix" and has_data(c):
                    return A.fig_fix3d(c, b, self.t0)
                if c.kind == "cloud":
                    return A.fig_cloud3d(c, b)
                if c.kind == "tf":
                    return A.fig_tf3d(c)
                if c.kind == "imu" and has_data(c):
                    return A.fig_imu3d({n: c}, {n: A.imu_stats(c)})
                raise ValueError(f"No hay vista 3D para {c.type}.")
            self.show(build, f"3D — {n}")

    def plot_timing(self) -> None:
        for n in self._selected():
            def build(n=n):
                c, _ = self.view(n)
                return A.fig_timing(c, self.t0)
            self.show(build, f"Temporización — {n}")

    def plot_global(self) -> None:
        if not self._need_bag():
            return
        self.show(lambda: A.fig_global(self.col, self.st, self.t0),
                  "Tasas y ancho de banda")

    def export_csv(self) -> None:
        sel = self._selected()
        if not sel:
            return
        prefix = filedialog.asksaveasfilename(
            title="Prefijo del CSV (se añade _topic.csv)", initialfile="bag")
        if not prefix:
            return
        try:
            out = ""
            for n in sel:
                c, _ = self.view(n)
                _, text = self.captured(A.export_csv, c, prefix)
                out += text
            messagebox.showinfo("CSV exportado", out.strip() or
                                "Esos topics no tienen series numéricas exportables.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"{type(exc).__name__}: {exc}")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    App(path).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
