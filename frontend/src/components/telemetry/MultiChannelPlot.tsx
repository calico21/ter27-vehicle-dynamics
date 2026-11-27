import React from "react";
import type { TelemetryFrame } from "../../types";

interface MultiChannelPlotProps {
  history: TelemetryFrame[];
}

export function MultiChannelPlot({ history }: MultiChannelPlotProps) {
  const latest = history[history.length - 1];

  return (
    <section className="panel">
      <div className="panel-header">
        <h3 className="panel-title">Multi-Channel Time Series (60 Hz Buffer)</h3>
        <span className="badge">{history.length} frames</span>
      </div>

      <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: 0 }}>
        Synchronized channels: vx (m/s), ay (g), omega_z (rad/s), steer (deg), and EKF estimate
        overlay. Expand with Recharts ResponsiveContainer or Canvas in Google AI Studio.
      </p>

      {latest ? (
        <table className="data-table">
          <thead>
            <tr>
              <th>Channel</th>
              <th>Truth (14-DOF)</th>
              <th>EKF Estimate</th>
              <th>Unit</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>vx</td>
              <td>{latest.state.vx.toFixed(3)}</td>
              <td>{latest.ekf.ekf_vx.toFixed(3)}</td>
              <td>m/s</td>
            </tr>
            <tr>
              <td>vy</td>
              <td>{latest.state.vy.toFixed(3)}</td>
              <td>{latest.ekf.ekf_vy.toFixed(3)}</td>
              <td>m/s</td>
            </tr>
            <tr>
              <td>omega_z</td>
              <td>{latest.state.yaw_rate_rad_s.toFixed(3)}</td>
              <td>{latest.ekf.ekf_yaw_rate.toFixed(3)}</td>
              <td>rad/s</td>
            </tr>
            <tr>
              <td>beta</td>
              <td>{latest.state.beta_deg.toFixed(2)}</td>
              <td>{latest.ekf.ekf_beta_deg.toFixed(2)}</td>
              <td>deg</td>
            </tr>
          </tbody>
        </table>
      ) : null}
    </section>
  );
}
