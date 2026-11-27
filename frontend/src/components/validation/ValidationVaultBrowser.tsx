import React, { useEffect, useState } from "react";
import type { VaultCatalogResponse } from "../../types";
import { fetchVaultCatalog } from "../../services/apiClient";

export function ValidationVaultBrowser() {
  const [catalog, setCatalog] = useState<VaultCatalogResponse | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);

  useEffect(() => {
    let active = true;
    fetchVaultCatalog().then((res) => {
      if (!active) return;
      setCatalog(res.data);
      setIsLive(res.live);
    });
    return () => {
      active = false;
    };
  }, []);

  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">Validation Vault Catalog (Suites 01..10)</h2>
        <span className={isLive ? "badge live" : "badge gold"}>
          {isLive ? "LIVE API" : "OFFLINE MOCK"}
        </span>
      </div>

      {catalog ? (
        <>
          <div className="grid-4" style={{ marginBottom: "14px" }}>
            <div className="kpi-card">
              <div className="kpi-label">Suites</div>
              <div className="kpi-value">{catalog.n_suites}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">300-DPI Figures</div>
              <div className="kpi-value">{catalog.total_pngs}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Metric JSONs</div>
              <div className="kpi-value">{catalog.total_jsons}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">Total PNG Size</div>
              <div className="kpi-value">{catalog.total_png_size_mb.toFixed(1)} MB</div>
            </div>
          </div>

          <table className="data-table">
            <thead>
              <tr>
                <th>Code</th>
                <th>Suite Title</th>
                <th>Tag</th>
                <th>Figures</th>
                <th>Key KPIs</th>
              </tr>
            </thead>
            <tbody>
              {catalog.suites.map((suite) => (
                <tr key={suite.code}>
                  <td>{suite.code}</td>
                  <td>{suite.title}</td>
                  <td>{suite.tag}</td>
                  <td>{suite.n_figures}</td>
                  <td>
                    {suite.kpis
                      .slice(0, 2)
                      .map((k) => `${k.label}: ${k.value}`)
                      .join(" | ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : null}
    </section>
  );
}
