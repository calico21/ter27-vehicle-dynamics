import React from "react";
import type { SessionABComparisonResponse } from "../../types";

interface SectorDeltaBreakdownProps {
  comparison: SessionABComparisonResponse;
}

export function SectorDeltaBreakdown({ comparison }: SectorDeltaBreakdownProps) {
  const driverSamples = comparison.driver_trace;
  const refSamples = comparison.casadi_reference_trace;
  const previewRows = driverSamples.slice(0, 8);

  return (
    <section className="panel">
      <div className="panel-header">
        <h3 className="panel-title">Trace &amp; Sector Delta Breakdown</h3>
        <span className="badge">
          Driver: {driverSamples.length} pts | Ref: {refSamples.length} pts
        </span>
      </div>

      <table className="data-table">
        <thead>
          <tr>
            <th>s (m)</th>
            <th>t (s)</th>
            <th>vx (m/s)</th>
            <th>ay (g)</th>
            <th>beta (deg)</th>
            <th>Mz_TV (Nm)</th>
            <th>delta_t (s)</th>
          </tr>
        </thead>
        <tbody>
          {previewRows.map((pt, idx) => (
            <tr key={idx}>
              <td>{pt.s_m.toFixed(1)}</td>
              <td>{pt.t_s.toFixed(2)}</td>
              <td>{pt.vx_mps.toFixed(2)}</td>
              <td>{pt.ay_g.toFixed(2)}</td>
              <td>{pt.beta_deg.toFixed(2)}</td>
              <td>{pt.mz_tv_nm.toFixed(1)}</td>
              <td>{pt.delta_t_s.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
