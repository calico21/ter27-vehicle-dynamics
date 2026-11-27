import React, { useEffect, useState } from "react";
import type { SessionABComparisonResponse, SessionSummary } from "../../types";
import { fetchSessionComparison, fetchSessionsList } from "../../services/apiClient";
import { SectorDeltaBreakdown } from "./SectorDeltaBreakdown";

export function SessionABComparator() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [comparison, setComparison] = useState<SessionABComparisonResponse | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);

  useEffect(() => {
    let active = true;
    fetchSessionsList().then((res) => {
      if (!active) return;
      setSessions(res.data.sessions);
      setIsLive(res.live);
      if (res.data.sessions.length > 0) {
        setSelectedId(res.data.sessions[0].session_id);
      }
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let active = true;
    fetchSessionComparison(selectedId).then((res) => {
      if (!active) return;
      setComparison(res.data);
      setIsLive(res.live);
    });
    return () => {
      active = false;
    };
  }, [selectedId]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <section className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Session A/B Forensics (DiL vs CasADi Reference)</h2>
          <span className={isLive ? "badge live" : "badge gold"}>
            {isLive ? "LIVE API" : "OFFLINE MOCK"}
          </span>
        </div>

        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginBottom: "12px" }}>
          {sessions.map((s) => (
            <button
              key={s.session_id}
              type="button"
              className={`btn ${s.session_id === selectedId ? "btn-gold" : ""}`}
              onClick={() => setSelectedId(s.session_id)}
            >
              {s.title}
            </button>
          ))}
        </div>

        {comparison ? (
          <div className="grid-4">
            {Object.entries(comparison.kpis)
              .slice(0, 8)
              .map(([key, val]) => (
                <div className="kpi-card" key={key}>
                  <div className="kpi-label">{key}</div>
                  <div className="kpi-value">{Number(val).toFixed(3)}</div>
                </div>
              ))}
          </div>
        ) : null}
      </section>

      {comparison ? <SectorDeltaBreakdown comparison={comparison} /> : null}
    </div>
  );
}
