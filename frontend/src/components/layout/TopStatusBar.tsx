import React from "react";
import { useTelemetryStore } from "../../state/telemetryStore";

export function TopStatusBar() {
  const { frame, connectionStatus } = useTelemetryStore();

  const bspdActive = frame.hv.bspd_active;
  const connBadgeClass =
    connectionStatus === "connected"
      ? "badge live"
      : connectionStatus === "reconnecting"
      ? "badge gold"
      : "badge alert";

  return (
    <header className="top-status-bar">
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <strong style={{ fontSize: "14px", letterSpacing: "0.03em" }}>
          TeR27 Vehicle Dynamics &amp; Telemetry Workbench
        </strong>
        <span className="badge gold">{frame.vehicle_id}</span>
        <span className="badge">{frame.grip_profile}</span>
        <span className="badge">{frame.sensor_profile}</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        <span className="badge">SRC: {frame.source}</span>
        <span className="badge">HV: {frame.hv.p_dc_kw.toFixed(1)} kW</span>
        <span className="badge">SOC: {frame.hv.soc_pct.toFixed(1)}%</span>
        <span className={bspdActive ? "badge alert" : "badge"}>
          BSPD: {bspdActive ? "TRIP" : "OK"}
        </span>
        <span className={connBadgeClass}>WS: {connectionStatus.toUpperCase()}</span>
      </div>
    </header>
  );
}
