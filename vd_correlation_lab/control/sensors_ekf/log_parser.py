#!/usr/bin/env python3
"""
Lector unificado de logs reales CAN (.log + TER.dbc), CSV y ROS bags.
Decodifica tramas SocketCAN bit a bit sin dependencias externas de cantools.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RE_BO = re.compile(r"^BO_\s+(\d+)\s+(\w+)\s*:\s*(\d+)\s+(\w+)")
RE_SG = re.compile(
    r"^SG_\s+(\w+)(?:\s+(M|m\d+))?\s*:\s*"
    r"(\d+)\|(\d+)@([01])([+-])\s*"
    r"\(([^,]+),([^)]+)\)\s*"
    r"\[([^|]+)\|([^\]]+)\]\s*"
    r'"([^"]*)"'
)
RE_CANDUMP = re.compile(r"^\(([\d.]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)$")

CAN_SIGNAL_ALIASES: Dict[str, List[str]] = {
    "front_vx_av_ms": ["Front_V.vx_av", "FRONT_V.vx_av"],
    "wheel_speed_kmh": ["WheelInfo.speed", "WHEEL_INFO.speed"],
    "wheel_rpm_fl": ["Front_RPM.RPM_L", "WheelInfo.fl_rpm", "WHEEL_INFO.fl_rpm"],
    "wheel_rpm_fr": ["Front_RPM.RPM_R", "WheelInfo.fr_rpm", "WHEEL_INFO.fr_rpm"],
    "wheel_rpm_rl": ["WheelInfo.rlRPM", "WheelInfo.rl_rpm", "WHEEL_INFO.rl_rpm"],
    "wheel_rpm_rr": ["WheelInfo.rrRPM", "WheelInfo.rr_rpm", "WHEEL_INFO.rr_rpm"],
    "imu_ax_raw": ["Accel.a_x", "IMU_ACCEL.AccelX", "Accel.ax", "ACCEL.ax"],
    "imu_ay_raw": ["Accel.a_y", "IMU_ACCEL.AccelY", "Accel.ay", "ACCEL.ay"],
    "imu_az_raw": ["Accel.a_z", "IMU_ACCEL.AccelZ", "Accel.az", "ACCEL.az"],
    "imu_wz_degs": ["AngRate.Yaw_Rate_z", "IMU_GYRO.GyroZ", "AngRate.wz", "ANG_RATE.wz"],
    "apps_av_raw": ["APPS.APPS_AV", "PEDALS.apps_av"],
    "bpps_bar": ["BPPS.BPPS", "PEDALS.bpps"],
    "steer_raw_deg": ["STEER.ANGLE", "STEERING.angle"],
    "susp_fl_raw": ["Front_Susp.flDisp", "SUSP_F.fl", "SUSP.fl"],
    "susp_fr_raw": ["Front_Susp.frDisp", "SUSP_F.fr", "SUSP.fr"],
    "susp_rl_raw": ["Rear_Susp.rlDisp", "SUSP_R.rl", "SUSP.rl"],
    "susp_rr_raw": ["Rear_Susp.rrDisp", "SUSP_R.rr", "SUSP.rr"],
}


@dataclass
class DbcSignal:
    name: str
    start_bit: int
    length: int
    byte_order: int
    signed: bool
    factor: float
    offset: float
    min_val: float
    max_val: float
    unit: str
    is_muxor: bool = False
    mux_val: Optional[int] = None


@dataclass
class DbcMessage:
    can_id: int
    raw_id: int
    name: str
    dlc: int
    sender: str
    signals: List[DbcSignal] = field(default_factory=list)
    muxor: Optional[DbcSignal] = None


class DbcDatabase:
    """Base de datos DBC ligera para decodificación bit a bit de SocketCAN."""

    def __init__(self, dbc_path: Path) -> None:
        self.dbc_path = Path(dbc_path)
        self.messages: Dict[int, DbcMessage] = {}
        self._load(self.dbc_path)

    def _load(self, path: Path) -> None:
        current_msg: Optional[DbcMessage] = None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                m_bo = RE_BO.match(line)
                if m_bo:
                    raw_id = int(m_bo.group(1))
                    can_id = raw_id & 0x1FFFFFFF
                    current_msg = DbcMessage(
                        can_id=can_id,
                        raw_id=raw_id,
                        name=m_bo.group(2),
                        dlc=int(m_bo.group(3)),
                        sender=m_bo.group(4),
                    )
                    self.messages[can_id] = current_msg
                    continue
                if current_msg is not None and line.startswith("SG_ "):
                    m_sg = RE_SG.match(line)
                    if m_sg:
                        mux_tok = m_sg.group(2)
                        sig = DbcSignal(
                            name=m_sg.group(1),
                            start_bit=int(m_sg.group(3)),
                            length=int(m_sg.group(4)),
                            byte_order=int(m_sg.group(5)),
                            signed=(m_sg.group(6) == "-"),
                            factor=float(m_sg.group(7)),
                            offset=float(m_sg.group(8)),
                            min_val=float(m_sg.group(9)),
                            max_val=float(m_sg.group(10)),
                            unit=m_sg.group(11),
                            is_muxor=(mux_tok == "M"),
                            mux_val=int(mux_tok[1:]) if (mux_tok and mux_tok.startswith("m")) else None,
                        )
                        current_msg.signals.append(sig)
                        if sig.is_muxor:
                            current_msg.muxor = sig

    @staticmethod
    def _extract_bits(payload: bytes, start_bit: int, length: int, byte_order: int, signed: bool) -> int:
        nbytes = len(payload)
        raw = 0
        if byte_order == 1:  # intel (little-endian)
            for i in range(length):
                bit_idx = start_bit + i
                byte_idx = bit_idx // 8
                if byte_idx < nbytes:
                    bit_in_byte = bit_idx % 8
                    if (payload[byte_idx] >> bit_in_byte) & 1:
                        raw |= 1 << i
        else:  # motorola (big-endian)
            bit_pos = start_bit
            for i in range(length):
                byte_idx = bit_pos // 8
                bit_in_byte = bit_pos % 8
                bit_val = (payload[byte_idx] >> bit_in_byte) & 1 if byte_idx < nbytes else 0
                raw = (raw << 1) | bit_val
                if bit_in_byte == 0:
                    bit_pos = (byte_idx + 1) * 8 + 7
                else:
                    bit_pos -= 1
        if signed and (raw & (1 << (length - 1))):
            raw -= 1 << length
        return raw

    def decode_frame(self, can_id: int, payload: bytes) -> Dict[str, float]:
        msg = self.messages.get(can_id)
        if msg is None:
            return {}
        out: Dict[str, float] = {}
        mux_val: Optional[int] = None
        if msg.muxor is not None:
            raw_m = self._extract_bits(payload, msg.muxor.start_bit, msg.muxor.length, msg.muxor.byte_order, False)
            mux_val = int(round(raw_m * msg.muxor.factor + msg.muxor.offset))
            out[f"{msg.name}.{msg.muxor.name}"] = float(mux_val)
        for sig in msg.signals:
            if sig.is_muxor:
                continue
            if sig.mux_val is not None and sig.mux_val != mux_val:
                continue
            raw = self._extract_bits(payload, sig.start_bit, sig.length, sig.byte_order, sig.signed)
            out[f"{msg.name}.{sig.name}"] = raw * sig.factor + sig.offset
        return out


def parse_candump_log(log_path: Path, dbc: DbcDatabase) -> Tuple[Dict[str, Tuple[List[float], List[float]]], float]:
    """Decodifica un archivo candump .log en flujos temporales por señal."""
    streams: Dict[str, Tuple[List[float], List[float]]] = {}
    t0: Optional[float] = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_CANDUMP.match(line.strip())
            if not m:
                continue
            ts = float(m.group(1))
            if t0 is None:
                t0 = ts
            t_rel = ts - t0
            can_id = int(m.group(2), 16) & 0x1FFFFFFF
            hex_data = m.group(3)
            if len(hex_data) % 2 == 1:
                hex_data = "0" + hex_data
            payload = bytes.fromhex(hex_data) if hex_data else b""
            decoded = dbc.decode_frame(can_id, payload)
            for k, v in decoded.items():
                if k not in streams:
                    streams[k] = ([], [])
                streams[k][0].append(t_rel)
                streams[k][1].append(float(v))
    t_end = 0.0
    for ts_list, _ in streams.values():
        if ts_list:
            t_end = max(t_end, ts_list[-1])
    return streams, t_end


def load_log_to_grid(
    log_path: Path,
    dbc: Optional[DbcDatabase],
    fs_hz: float = 100.0,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[List[float], List[float]]], Dict[str, int], bool]:
    """Carga un .log CAN o .csv a una rejilla uniforme a fs_hz.
    Devuelve (df, raw_streams, present_counts, is_preconditioned_csv).
    """
    ext = log_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(log_path)
        return df, {}, {c: len(df) for c in df.columns}, True

    if dbc is None:
        raise ValueError(f"Se requiere TER.dbc para decodificar {log_path}")

    streams, t_end = parse_candump_log(log_path, dbc)
    if t_end <= 0.05:
        raise ValueError(f"Log vacío o sin tramas válidas: {log_path}")

    dt = 1.0 / fs_hz
    t_grid = np.arange(0.0, t_end, dt, dtype=np.float64)
    df = pd.DataFrame({"t_s": t_grid})
    present_counts: Dict[str, int] = {}

    for col, aliases in CAN_SIGNAL_ALIASES.items():
        st = None
        for alias in aliases:
            if alias in streams and len(streams[alias][0]) >= 2:
                st = streams[alias]
                break
        if st is None:
            df[col] = np.nan
            present_counts[col] = 0
        else:
            df[col] = np.interp(t_grid, np.asarray(st[0], dtype=np.float64), np.asarray(st[1], dtype=np.float64))
            present_counts[col] = len(st[0])

    return df, streams, present_counts, False
