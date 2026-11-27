"""Real-time UDP <-> WebSocket telemetry bridge with DiL/synthetic 60 Hz fallback."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import socket
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from api.core.config import (
    DIL_SESSIONS_DIR,
    UDP_COMMAND_HOST,
    UDP_COMMAND_PORT,
    UDP_LIVE_TIMEOUT_S,
    UDP_TELEMETRY_HOST,
    UDP_TELEMETRY_PORT,
)
from api.models.telemetry import (
    DilDeltaTelemetry,
    EkfStateTelemetry,
    HvPowerTelemetry,
    TelemetryFrame,
    TelemetrySource,
    VehicleState14Dof,
    WheelsTelemetry,
    WheelTelemetry,
)


class _UdpTelemetryProtocol(asyncio.DatagramProtocol):
    def __init__(self, bridge: "TelemetryBridge") -> None:
        self._bridge = bridge

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        try:
            payload = json.loads(data.decode("utf-8"))
            if isinstance(payload, dict):
                self._bridge.ingest_udp_payload(payload)
        except Exception:
            pass

    def error_received(self, exc: Exception) -> None:
        pass


class TelemetryBridge:
    """Listens on UDP 9876, sends commands to UDP 9877, and streams 60 Hz frames."""

    def __init__(self) -> None:
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._last_udp_frame: Optional[TelemetryFrame] = None
        self._last_udp_rx_mono: float = 0.0
        self._ws_clients: Set[Any] = set()
        self._start_mono: float = time.monotonic()
        self._dil_cache: List[Dict[str, float]] = []
        self._dil_cache_mtime: float = 0.0
        self._dil_cursor: int = 0

    async def start(self) -> None:
        if self._transport is not None:
            return
        loop = asyncio.get_running_loop()
        try:
            transport, _ = await loop.create_datagram_endpoint(
                lambda: _UdpTelemetryProtocol(self),
                local_addr=(UDP_TELEMETRY_HOST, UDP_TELEMETRY_PORT),
            )
            self._transport = transport
        except OSError:
            # Port may already be bound by another worker or test instance.
            self._transport = None

    async def stop(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def register_ws(self, ws: Any) -> None:
        self._ws_clients.add(ws)

    def unregister_ws(self, ws: Any) -> None:
        self._ws_clients.discard(ws)

    @property
    def active_ws_clients(self) -> int:
        return len(self._ws_clients)

    @property
    def is_udp_connected(self) -> bool:
        if self._last_udp_frame is None:
            return False
        return (time.monotonic() - self._last_udp_rx_mono) <= UDP_LIVE_TIMEOUT_S

    @property
    def active_source(self) -> TelemetrySource:
        if self.is_udp_connected:
            return "godot_udp"
        if self._has_dil_recording():
            return "dil_session"
        return "synthetic_fallback"

    def ingest_udp_payload(self, raw: Dict[str, Any]) -> None:
        try:
            raw_copy = dict(raw)
            raw_copy["source"] = "godot_udp"
            frame = TelemetryFrame.model_validate(raw_copy)
            self._last_udp_frame = frame
            self._last_udp_rx_mono = time.monotonic()
        except Exception:
            pass

    def send_godot_command(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """Non-blocking UDP JSON dispatch to MirenaSim command listener (127.0.0.1:9877)."""
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setblocking(False)
            sock.sendto(data, (UDP_COMMAND_HOST, UDP_COMMAND_PORT))
            return True, "UDP command dispatched to MirenaSim listener"
        except OSError as exc:
            return False, f"UDP send warning: {exc}"
        finally:
            sock.close()

    def get_next_frame(self) -> TelemetryFrame:
        if self.is_udp_connected and self._last_udp_frame is not None:
            return self._last_udp_frame
        dil_frame = self._next_dil_session_frame()
        if dil_frame is not None:
            return dil_frame
        return self._synthesize_14dof_frame()

    def _has_dil_recording(self) -> bool:
        if not DIL_SESSIONS_DIR.exists():
            return False
        for path in sorted(DIL_SESSIONS_DIR.glob("*.csv")):
            if path.is_file() and path.stat().st_size > 128:
                return True
        return False

    def _load_latest_dil_csv(self) -> List[Dict[str, float]]:
        if not DIL_SESSIONS_DIR.exists():
            return []
        csv_files = sorted(
            [p for p in DIL_SESSIONS_DIR.glob("*.csv") if p.is_file() and p.stat().st_size > 128],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not csv_files:
            return []
        latest = csv_files[0]
        mtime = latest.stat().st_mtime
        if self._dil_cache and mtime == self._dil_cache_mtime:
            return self._dil_cache
        rows: List[Dict[str, float]] = []
        try:
            with latest.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    parsed: Dict[str, float] = {}
                    for k, v in row.items():
                        if k is None or v is None:
                            continue
                        try:
                            parsed[k.strip()] = float(v)
                        except ValueError:
                            continue
                    if parsed:
                        rows.append(parsed)
        except Exception:
            return []
        self._dil_cache = rows
        self._dil_cache_mtime = mtime
        self._dil_cursor = 0
        return rows

    def _next_dil_session_frame(self) -> Optional[TelemetryFrame]:
        rows = self._load_latest_dil_csv()
        if not rows:
            return None
        row = rows[self._dil_cursor % len(rows)]
        self._dil_cursor = (self._dil_cursor + 1) % len(rows)

        t_s = row.get("t_s", 0.0)
        vx = row.get("vx_ms", 12.0)
        ay_mps2 = row.get("ay_mps2", 0.0)
        ax_mps2 = row.get("ax_mps2", 0.0)
        wz = row.get("wz_rads", 0.0)
        steer_deg = math.degrees(row.get("delta_rad", 0.0))
        brake = max(0.0, min(1.0, row.get("brake_in", 0.0)))
        throttle = max(0.0, min(1.0, 0.75 if ax_mps2 >= 0.0 and brake < 0.05 else 0.0))
        ay_g = ay_mps2 / 9.81
        ax_g = ax_mps2 / 9.81
        beta_deg = math.degrees(math.atan2(-0.04 * ay_g, max(abs(vx), 1.0)))

        wheels = self._compute_wheels_from_accel(vx, ax_g, ay_g, wz, steer_deg, t_s)
        p_dc_kw = max(0.0, min(80.0, 290.0 * max(ax_mps2, 0.0) * vx * 0.001 + 4.5))
        p_regen_kw = max(0.0, min(25.0, 290.0 * max(-ax_mps2, 0.0) * vx * 0.0005))

        return TelemetryFrame(
            timestamp_s=round(t_s, 4),
            source="dil_session",
            vehicle_id="ter27_4wd",
            grip_profile="FSG_RUBBERED",
            sensor_profile="SENSOR_TRACK_REALISM",
            state=VehicleState14Dof(
                vx=round(vx, 4),
                vy=round(vx * math.tan(math.radians(beta_deg)), 4),
                vz=0.0,
                ax_g=round(ax_g, 4),
                ay_g=round(ay_g, 4),
                yaw_rate_rad_s=round(wz, 4),
                beta_deg=round(beta_deg, 4),
                steer_deg=round(steer_deg, 4),
                throttle=round(throttle, 4),
                brake=round(brake, 4),
                x_m=round(row.get("s_m", 0.0), 3),
                y_m=round(row.get("e_lat_m", 0.0), 3),
                z_m=0.29,
                heading_rad=round(wz * 0.05, 4),
            ),
            wheels=wheels,
            hv=HvPowerTelemetry(
                p_dc_kw=round(p_dc_kw, 3),
                p_regen_kw=round(p_regen_kw, 3),
                soc_pct=92.4,
                bspd_active=bool(brake > 0.15 and p_dc_kw > 5.0),
                tv_mz_nm=round(180.0 * ay_g, 2),
            ),
            ekf=EkfStateTelemetry(
                ekf_vx=round(vx + 0.03 * math.sin(t_s * 7.0), 4),
                ekf_vy=round(
                    vx * math.tan(math.radians(beta_deg)) + 0.02 * math.cos(t_s * 5.0), 4
                ),
                ekf_yaw_rate=round(wz + 0.005 * math.sin(t_s * 9.0), 4),
                ekf_beta_deg=round(beta_deg + 0.08 * math.sin(t_s * 5.0), 4),
                sensor_latency_ms=125.0,
            ),
            dil_delta=DilDeltaTelemetry(
                delta_time_s=round(row.get("delta_t_s", 0.0), 4),
                delta_vx_mps=round(row.get("delta_vx_ms", 0.0), 4),
                delta_ay_g=round((ay_mps2 - row.get("ay_ref_mps2", ay_mps2)) / 9.81, 4),
            ),
        )

    def _synthesize_14dof_frame(self) -> TelemetryFrame:
        """Synthesizes a physically coherent 14-DOF TeR27 Skidpad/Autocross frame at 60 Hz."""
        t_s = time.monotonic() - self._start_mono
        r_m = 8.40
        omega_orbit = 2.0 * math.pi / 4.61
        phase = t_s * omega_orbit
        circle_sign = 1.0 if (int(phase / (4.0 * math.pi)) % 2 == 0) else -1.0
        theta = circle_sign * (phase % (2.0 * math.pi))

        vx = 11.45 + 0.35 * math.sin(0.5 * phase)
        ay_mps2 = circle_sign * (vx * vx / r_m)
        ax_mps2 = 0.35 * 0.5 * omega_orbit * math.cos(0.5 * phase)
        ay_g = ay_mps2 / 9.81
        ax_g = ax_mps2 / 9.81
        wz = circle_sign * (vx / r_m)
        beta_deg = -circle_sign * (2.15 + 0.15 * math.sin(phase))
        vy = vx * math.tan(math.radians(beta_deg))
        steer_deg = circle_sign * (11.2 + 0.4 * math.sin(phase))
        throttle = 0.42 + 0.08 * max(0.0, math.cos(0.5 * phase))
        brake = 0.0

        cx = 0.0
        cy = circle_sign * 9.125
        x_m = cx + r_m * math.sin(theta)
        y_m = cy - circle_sign * r_m * math.cos(theta)
        heading_rad = theta % (2.0 * math.pi)

        wheels = self._compute_wheels_from_accel(vx, ax_g, ay_g, wz, steer_deg, t_s)
        p_dc_kw = 18.6 + 2.4 * math.sin(0.5 * phase)
        tv_mz_nm = circle_sign * (245.0 + 22.0 * math.cos(phase))

        return TelemetryFrame(
            timestamp_s=round(t_s, 4),
            source="synthetic_fallback",
            vehicle_id="ter27_4wd",
            grip_profile="FSG_RUBBERED",
            sensor_profile="SENSOR_TRACK_REALISM",
            state=VehicleState14Dof(
                vx=round(vx, 4),
                vy=round(vy, 4),
                vz=0.0,
                ax_g=round(ax_g, 4),
                ay_g=round(ay_g, 4),
                yaw_rate_rad_s=round(wz, 4),
                beta_deg=round(beta_deg, 4),
                steer_deg=round(steer_deg, 4),
                throttle=round(throttle, 4),
                brake=brake,
                x_m=round(x_m, 4),
                y_m=round(y_m, 4),
                z_m=0.29,
                heading_rad=round(heading_rad, 4),
            ),
            wheels=wheels,
            hv=HvPowerTelemetry(
                p_dc_kw=round(p_dc_kw, 3),
                p_regen_kw=0.0,
                soc_pct=round(max(10.0, 95.0 - 0.015 * t_s), 3),
                bspd_active=False,
                tv_mz_nm=round(tv_mz_nm, 2),
            ),
            ekf=EkfStateTelemetry(
                ekf_vx=round(vx + 0.02 * math.sin(t_s * 6.0), 4),
                ekf_vy=round(vy + 0.015 * math.cos(t_s * 4.0), 4),
                ekf_yaw_rate=round(wz + 0.004 * math.sin(t_s * 8.0), 4),
                ekf_beta_deg=round(beta_deg + 0.07 * math.cos(t_s * 4.0), 4),
                sensor_latency_ms=125.0,
            ),
            dil_delta=DilDeltaTelemetry(
                delta_time_s=round(-0.017 + 0.008 * math.sin(phase), 4),
                delta_vx_mps=round(0.12 * math.cos(phase), 4),
                delta_ay_g=round(0.03 * math.sin(phase), 4),
            ),
        )

    @staticmethod
    def _compute_wheels_from_accel(
        vx: float,
        ax_g: float,
        ay_g: float,
        wz: float,
        steer_deg: float,
        t_s: float,
    ) -> WheelsTelemetry:
        mg = 290.0 * 9.81
        f_down = 0.5 * 1.225 * 6.0 * 0.78 * vx * vx
        dfz_long = 290.0 * (ax_g * 9.81) * 0.29 / 1.54
        fz_f_tot = max(100.0, mg * 0.45 + f_down * 0.48 - dfz_long)
        fz_r_tot = max(100.0, mg * 0.55 + f_down * 0.52 + dfz_long)
        dfz_lat_f = 290.0 * (ay_g * 9.81) * 0.29 * 0.50 / 1.23
        dfz_lat_r = 290.0 * (ay_g * 9.81) * 0.29 * 0.50 / 1.23

        fz_vals = [
            max(40.0, 0.5 * fz_f_tot - dfz_lat_f),
            max(40.0, 0.5 * fz_f_tot + dfz_lat_f),
            max(40.0, 0.5 * fz_r_tot - dfz_lat_r),
            max(40.0, 0.5 * fz_r_tot + dfz_lat_r),
        ]
        camber_base = [-0.82, 0.82, -1.32, 1.32]
        roll_deg = ay_g * 0.80
        wheels_out: List[WheelTelemetry] = []
        for idx in range(4):
            fz = fz_vals[idx]
            fy = fz * ay_g * 0.96
            fx = fz * ax_g * 0.90 + (110.0 if idx in (1, 3) else -110.0) * (
                1.0 if ay_g >= 0.0 else -1.0
            )
            f_comb = math.hypot(fx, fy)
            util = min(1.25, f_comb / max(fz * 1.65, 1.0))
            alpha_deg = 5.4 * (ay_g / 1.5) * (1.05 if idx < 2 else 0.95)
            kappa = 0.025 * ax_g + (0.012 if idx >= 2 else 0.008)
            camber_deg = camber_base[idx] - roll_deg
            travel_m = (fz - 710.0) / 27400.0
            torque_nm = fx * 0.2032
            temp_c = 74.0 + 3.5 * util + 0.8 * math.sin(0.2 * t_s + idx)
            wheels_out.append(
                WheelTelemetry(
                    fx_n=round(fx, 2),
                    fy_n=round(fy, 2),
                    fz_n=round(fz, 2),
                    slip_angle_deg=round(alpha_deg, 3),
                    slip_ratio=round(kappa, 4),
                    tire_util=round(util, 4),
                    camber_deg=round(camber_deg, 3),
                    wheel_travel_m=round(travel_m, 5),
                    motor_torque_nm=round(torque_nm, 2),
                    temp_c=round(temp_c, 2),
                )
            )
        return WheelsTelemetry(
            fl=wheels_out[0],
            fr=wheels_out[1],
            rl=wheels_out[2],
            rr=wheels_out[3],
        )


telemetry_bridge = TelemetryBridge()
