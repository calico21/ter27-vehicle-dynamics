import React from "react";
import type { SuspensionOverview } from "../../types";

interface SuspensionKinematicsOverlayProps {
  suspension: SuspensionOverview;
}

export function SuspensionKinematicsOverlay({ suspension }: SuspensionKinematicsOverlayProps) {
  const fl = suspension.front_left_hardpoints;

  return (
    <section className="panel">
      <div className="panel-header">
        <h3 className="panel-title">3D Double-Wishbone Kinematics &amp; Hardpoints</h3>
        <span className="badge">{suspension.frame}</span>
      </div>

      <div className="grid-4" style={{ marginBottom: "12px" }}>
        <div className="kpi-card">
          <div className="kpi-label">LLTD Elastic Front</div>
          <div className="kpi-value">{(suspension.lltd_elastic_front * 100).toFixed(1)}%</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Static Camber F/R</div>
          <div className="kpi-value">
            {suspension.static_camber_front_deg.toFixed(2)} /{" "}
            {suspension.static_camber_rear_deg.toFixed(2)} deg
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Motion Ratio F/R</div>
          <div className="kpi-value">
            {suspension.motion_ratio_front.toFixed(3)} / {suspension.motion_ratio_rear.toFixed(3)}
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">Caster / KPI</div>
          <div className="kpi-value">
            {suspension.caster_deg.toFixed(1)} / {suspension.kpi_deg.toFixed(1)} deg
          </div>
        </div>
      </div>

      <table className="data-table">
        <thead>
          <tr>
            <th>FL Hardpoint</th>
            <th>X (m)</th>
            <th>Y (m)</th>
            <th>Z (m)</th>
          </tr>
        </thead>
        <tbody>
          {(
            [
              ["LCA Front", fl.lca_front],
              ["LCA Rear", fl.lca_rear],
              ["LBJ", fl.lbj],
              ["UCA Front", fl.uca_front],
              ["UCA Rear", fl.uca_rear],
              ["UBJ", fl.ubj],
              ["Wheel Center", fl.wheel_center],
              ["Contact Patch", fl.contact_patch],
            ] as Array<[string, number[]]>
          ).map(([name, coords]) => (
            <tr key={name}>
              <td>{name}</td>
              <td>{coords[0]?.toFixed(4)}</td>
              <td>{coords[1]?.toFixed(4)}</td>
              <td>{coords[2]?.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
