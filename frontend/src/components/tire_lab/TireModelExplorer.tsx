import React, { useEffect, useState } from "react";
import type {
  GripProfileName,
  TireCompoundName,
  TireEvaluateRequest,
  TireEvaluateResponse,
} from "../../types";
import { evaluateTireModel } from "../../services/apiClient";

export function TireModelExplorer() {
  const [params, setParams] = useState<Required<TireEvaluateRequest>>({
    compound: "R25B",
    grip_profile: "FSG_RUBBERED",
    fz_n: 750,
    camber_deg: -1.5,
    temp_c: 75,
    pressure_kpa: 83,
    n_points: 61,
  });
  const [result, setResult] = useState<TireEvaluateResponse | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);

  useEffect(() => {
    let active = true;
    evaluateTireModel(params).then((res) => {
      if (!active) return;
      setResult(res.data);
      setIsLive(res.live);
    });
    return () => {
      active = false;
    };
  }, [params]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">MF 6.2 Parametric Tire Lab (Hoosier 16.0x7.5-10)</h2>
        <span className={isLive ? "badge live" : "badge gold"}>
          {isLive ? "LIVE API" : "OFFLINE MOCK"}
        </span>
      </div>

      <div className="grid-4" style={{ marginBottom: "14px" }}>
        <label className="kpi-card">
          <div className="kpi-label">Compound</div>
          <select
            className="btn"
            style={{ width: "100%" }}
            value={params.compound}
            onChange={(e) =>
              setParams((p) => ({ ...p, compound: e.target.value as TireCompoundName }))
            }
          >
            <option value="R25B">R25B</option>
            <option value="R20">R20</option>
          </select>
        </label>

        <label className="kpi-card">
          <div className="kpi-label">Grip Profile</div>
          <select
            className="btn"
            style={{ width: "100%" }}
            value={params.grip_profile}
            onChange={(e) =>
              setParams((p) => ({ ...p, grip_profile: e.target.value as GripProfileName }))
            }
          >
            <option value="FSG_RUBBERED">FSG_RUBBERED (0.57)</option>
            <option value="COLD_TEST_OCT">COLD_TEST_OCT (0.47)</option>
            <option value="FSG_REAL_TRACK">FSG_REAL_TRACK</option>
            <option value="TTC_LAB_PEAK">TTC_LAB_PEAK</option>
          </select>
        </label>

        <label className="kpi-card">
          <div className="kpi-label">Normal Load Fz ({params.fz_n} N)</div>
          <input
            type="range"
            min={200}
            max={2000}
            step={50}
            value={params.fz_n}
            onChange={(e) => setParams((p) => ({ ...p, fz_n: Number(e.target.value) }))}
            style={{ width: "100%" }}
          />
        </label>

        <label className="kpi-card">
          <div className="kpi-label">Tire Temp ({params.temp_c} C)</div>
          <input
            type="range"
            min={25}
            max={105}
            step={5}
            value={params.temp_c}
            onChange={(e) => setParams((p) => ({ ...p, temp_c: Number(e.target.value) }))}
            style={{ width: "100%" }}
          />
        </label>
      </div>

      {result ? (
        <div className="grid-4">
          <div className="kpi-card">
            <div className="kpi-label">Peak Fy</div>
            <div className="kpi-value">{result.peak_fy_n.toFixed(1)} N</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Peak Fx</div>
            <div className="kpi-value">{result.peak_fx_n.toFixed(1)} N</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Effective mu_y / mu_x</div>
            <div className="kpi-value">
              {result.mu_y_eff.toFixed(3)} / {result.mu_x_eff.toFixed(3)}
            </div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Thermal Factor</div>
            <div className="kpi-value">{result.thermal_mu_factor.toFixed(3)}</div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
