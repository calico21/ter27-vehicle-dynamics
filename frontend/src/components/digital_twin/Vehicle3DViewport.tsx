import React, { useEffect, useState } from "react";
import type { VehicleOverviewResponse } from "../../types";
import { fetchVehicleOverview } from "../../services/apiClient";
import { SuspensionKinematicsOverlay } from "./SuspensionKinematicsOverlay";

export function Vehicle3DViewport() {
  const [overview, setOverview] = useState<VehicleOverviewResponse | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);

  useEffect(() => {
    let active = true;
    fetchVehicleOverview().then((res) => {
      if (!active) return;
      setOverview(res.data);
      setIsLive(res.live);
    });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <section className="panel">
        <div className="panel-header">
          <h2 className="panel-title">TeR27 4WD 3D Digital Twin Viewport</h2>
          <span className={isLive ? "badge live" : "badge gold"}>
            {isLive ? "LIVE API" : "OFFLINE FALLBACK"}
          </span>
        </div>

        {overview ? (
          <div className="grid-4">
            <div className="kpi-card">
              <div className="kpi-label">Total Mass</div>
              <div className="kpi-value">{overview.params.mass_kg.toFixed(1)} kg</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Wheelbase</div>
              <div className="kpi-value">{overview.params.wheelbase_m.toFixed(3)} m</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">CG Height</div>
              <div className="kpi-value">{overview.params.cg_height_m.toFixed(3)} m</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">CL*A / CD*A</div>
              <div className="kpi-value">
                {overview.params.cl_a_m2.toFixed(2)} / {overview.params.cd_a_m2.toFixed(2)} m2
              </div>
            </div>
          </div>
        ) : (
          <div style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
            Loading TeR27 vehicle parameters and double-wishbone hardpoints...
          </div>
        )}
      </section>

      {overview ? <SuspensionKinematicsOverlay suspension={overview.suspension} /> : null}
    </div>
  );
}
