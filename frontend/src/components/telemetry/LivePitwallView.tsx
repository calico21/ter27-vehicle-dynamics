import React from "react";
import { useTelemetryStore } from "../../state/telemetryStore";
import { MultiChannelPlot } from "./MultiChannelPlot";
import { FrictionCircle2x2 } from "./FrictionCircle2x2";

export function LivePitwallView() {
  const { frame, history } = useTelemetryStore();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <section className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Live 14-DOF State &amp; Powertrain Telemetry</h2>
          <span className="mono" style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
            t = {frame.timestamp_s.toFixed(3)} s
          </span>
        </div>

        <div className="grid-4">
          <div className="kpi-card">
            <div className="kpi-label">Longitudinal Speed (vx)</div>
            <div className="kpi-value">{frame.state.vx.toFixed(2)} m/s</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Lateral Accel (ay)</div>
            <div className="kpi-value">{frame.state.ay_g.toFixed(2)} g</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Yaw Rate (omega_z)</div>
            <div className="kpi-value">{frame.state.yaw_rate_rad_s.toFixed(3)} rad/s</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">TV Yaw Moment (Mz)</div>
            <div className="kpi-value">{frame.hv.tv_mz_nm.toFixed(1)} Nm</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Sideslip (beta)</div>
            <div className="kpi-value">{frame.state.beta_deg.toFixed(2)} deg</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">EKF Sideslip (est)</div>
            <div className="kpi-value">{frame.ekf.ekf_beta_deg.toFixed(2)} deg</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">DiL Time Delta</div>
            <div className="kpi-value">
              {frame.dil_delta.delta_time_s >= 0 ? "+" : ""}
              {frame.dil_delta.delta_time_s.toFixed(3)} s
            </div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">CAN Latency</div>
            <div className="kpi-value">{frame.ekf.sensor_latency_ms.toFixed(1)} ms</div>
          </div>
        </div>
      </section>

      <div className="grid-2">
        <MultiChannelPlot history={history} />
        <FrictionCircle2x2 wheels={frame.wheels} />
      </div>
    </div>
  );
}
