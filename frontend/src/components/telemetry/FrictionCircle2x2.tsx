import React from "react";
import type { WheelsTelemetry, WheelTelemetry } from "../../types";

interface FrictionCircle2x2Props {
  wheels: WheelsTelemetry;
}

const CORNERS: Array<{ key: keyof WheelsTelemetry; label: string }> = [
  { key: "fl", label: "Front Left (FL)" },
  { key: "fr", label: "Front Right (FR)" },
  { key: "rl", label: "Rear Left (RL)" },
  { key: "rr", label: "Rear Right (RR)" },
];

function CornerCard({ label, wheel }: { label: string; wheel: WheelTelemetry }) {
  const saturated = wheel.tire_util >= 0.95;

  return (
    <div
      className="kpi-card"
      style={{
        borderColor: saturated ? "var(--accent-alert)" : "var(--border-subtle)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
        <span className="kpi-label" style={{ margin: 0 }}>
          {label}
        </span>
        <span className={saturated ? "badge alert" : "badge live"}>
          UTIL {(wheel.tire_util * 100).toFixed(0)}%
        </span>
      </div>
      <div className="mono" style={{ fontSize: "12px", lineHeight: 1.6 }}>
        <div>Fz: {wheel.fz_n.toFixed(0)} N | T: {wheel.temp_c.toFixed(1)} C</div>
        <div>Fx: {wheel.fx_n.toFixed(0)} N | Fy: {wheel.fy_n.toFixed(0)} N</div>
        <div>
          alpha: {wheel.slip_angle_deg.toFixed(2)} deg | kappa: {wheel.slip_ratio.toFixed(3)}
        </div>
        <div>Torque: {wheel.motor_torque_nm.toFixed(1)} Nm</div>
      </div>
    </div>
  );
}

export function FrictionCircle2x2({ wheels }: FrictionCircle2x2Props) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3 className="panel-title">4-Corner Tire Utilization &amp; Thermal Matrix</h3>
        <span className="badge gold">4WD In-Hub</span>
      </div>
      <div className="grid-2">
        {CORNERS.map(({ key, label }) => (
          <CornerCard key={key} label={label} wheel={wheels[key]} />
        ))}
      </div>
    </section>
  );
}
