import React, { useEffect, useState } from "react";
import type { SetupsMatrixResponse } from "../../types";
import { applySetupPreset, fetchSetupsMatrix } from "../../services/apiClient";
import { SimulationLauncher } from "./SimulationLauncher";

export function TracksideSetupMatrix() {
  const [matrix, setMatrix] = useState<SetupsMatrixResponse | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);
  const [applyStatus, setApplyStatus] = useState<string>("");

  useEffect(() => {
    let active = true;
    fetchSetupsMatrix().then((res) => {
      if (!active) return;
      setMatrix(res.data);
      setIsLive(res.live);
    });
    return () => {
      active = false;
    };
  }, []);

  const handleApply = async (presetName: string) => {
    setApplyStatus(`Dispatching ${presetName} to MirenaSim UDP 9877...`);
    const res = await applySetupPreset({ preset: presetName });
    setApplyStatus(res.data.message);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <section className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Trackside Setup Differential Matrix</h2>
          <span className={isLive ? "badge live" : "badge gold"}>
            {isLive ? "LIVE API" : "OFFLINE MOCK"}
          </span>
        </div>

        {applyStatus ? (
          <div className="badge gold" style={{ marginBottom: "12px" }}>
            {applyStatus}
          </div>
        ) : null}

        {matrix ? (
          <table className="data-table">
            <thead>
              <tr>
                <th>Preset</th>
                <th>Event</th>
                <th>LLTD</th>
                <th>Camber F/R (deg)</th>
                <th>Toe F/R (deg)</th>
                <th>Brake Bias</th>
                <th>Rake (mm)</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{matrix.base_setup.name}</td>
                <td>{matrix.base_setup.event}</td>
                <td>{matrix.base_setup.lltd.toFixed(3)}</td>
                <td>
                  {matrix.base_setup.camber_front_deg.toFixed(2)} /{" "}
                  {matrix.base_setup.camber_rear_deg.toFixed(2)}
                </td>
                <td>
                  {matrix.base_setup.toe_front_deg.toFixed(2)} /{" "}
                  {matrix.base_setup.toe_rear_deg.toFixed(2)}
                </td>
                <td>{matrix.base_setup.brake_bias.toFixed(3)}</td>
                <td>{matrix.base_setup.rake_mm.toFixed(1)}</td>
                <td>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => handleApply(matrix.base_setup.name)}
                  >
                    Apply
                  </button>
                </td>
              </tr>
              {matrix.presets.map((item) => (
                <tr key={item.preset.name}>
                  <td>{item.preset.name}</td>
                  <td>{item.preset.event}</td>
                  <td>{item.preset.lltd.toFixed(3)}</td>
                  <td>
                    {item.preset.camber_front_deg.toFixed(2)} /{" "}
                    {item.preset.camber_rear_deg.toFixed(2)}
                  </td>
                  <td>
                    {item.preset.toe_front_deg.toFixed(2)} /{" "}
                    {item.preset.toe_rear_deg.toFixed(2)}
                  </td>
                  <td>{item.preset.brake_bias.toFixed(3)}</td>
                  <td>{item.preset.rake_mm.toFixed(1)}</td>
                  <td>
                    <button
                      type="button"
                      className="btn btn-gold"
                      onClick={() => handleApply(item.preset.name)}
                    >
                      Apply
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </section>

      <SimulationLauncher />
    </div>
  );
}
