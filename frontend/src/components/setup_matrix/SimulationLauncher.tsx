import React, { useState } from "react";
import type {
  GripProfileName,
  SimulationEventName,
  SimulationJobStatusResponse,
  TireCompoundName,
} from "../../types";
import { launchSimulationJob, pollSimulationStatus } from "../../services/apiClient";

export function SimulationLauncher() {
  const [event, setEvent] = useState<SimulationEventName>("skidpad");
  const [gripProfile, setGripProfile] = useState<GripProfileName>("FSG_RUBBERED");
  const [compound, setCompound] = useState<TireCompoundName>("R25B");
  const [optimizeSetup, setOptimizeSetup] = useState<boolean>(true);
  const [dryRun, setDryRun] = useState<boolean>(true);
  const [job, setJob] = useState<SimulationJobStatusResponse | null>(null);

  const handleRun = async () => {
    const res = await launchSimulationJob({
      event,
      grip_profile: gripProfile,
      compound,
      optimize_setup: optimizeSetup,
      dry_run: dryRun,
    });
    setJob(res.data);
  };

  const handlePoll = async () => {
    if (!job?.job_id) return;
    const res = await pollSimulationStatus(job.job_id);
    setJob(res.data);
  };

  return (
    <section className="panel">
      <div className="panel-header">
        <h3 className="panel-title">CasADi Optimal Control Simulation Launcher</h3>
        <span className="badge">POST /api/simulation/run</span>
      </div>

      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "center" }}>
        <select
          className="btn"
          value={event}
          onChange={(e) => setEvent(e.target.value as SimulationEventName)}
        >
          <option value="skidpad">Skidpad</option>
          <option value="accel">Acceleration 75m</option>
          <option value="autocross">Autocross</option>
          <option value="endurance">Endurance 22km</option>
        </select>

        <select
          className="btn"
          value={gripProfile}
          onChange={(e) => setGripProfile(e.target.value as GripProfileName)}
        >
          <option value="FSG_RUBBERED">FSG_RUBBERED</option>
          <option value="COLD_TEST_OCT">COLD_TEST_OCT</option>
        </select>

        <select
          className="btn"
          value={compound}
          onChange={(e) => setCompound(e.target.value as TireCompoundName)}
        >
          <option value="R25B">R25B</option>
          <option value="R20">R20</option>
        </select>

        <label className="mono" style={{ fontSize: "12px" }}>
          <input
            type="checkbox"
            checked={optimizeSetup}
            onChange={(e) => setOptimizeSetup(e.target.checked)}
          />{" "}
          optimize_setup
        </label>

        <label className="mono" style={{ fontSize: "12px" }}>
          <input
            type="checkbox"
            checked={dryRun}
            onChange={(e) => setDryRun(e.target.checked)}
          />{" "}
          dry_run
        </label>

        <button type="button" className="btn btn-cyan" onClick={handleRun}>
          Launch Solver
        </button>

        {job ? (
          <button type="button" className="btn" onClick={handlePoll}>
            Refresh Status
          </button>
        ) : null}
      </div>

      {job ? (
        <div style={{ marginTop: "12px" }} className="kpi-card">
          <div className="mono" style={{ fontSize: "12px" }}>
            Job {job.job_id} | Status: {job.status} | Progress: {job.progress_pct.toFixed(0)}%
          </div>
          {job.log_tail.length > 0 ? (
            <pre
              className="mono"
              style={{
                marginTop: "8px",
                fontSize: "11px",
                color: "var(--text-secondary)",
                overflowX: "auto",
              }}
            >
              {job.log_tail.join("\n")}
            </pre>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
