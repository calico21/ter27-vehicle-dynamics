#!/usr/bin/env python3
"""
Residual NN trainer for the Mirena Dynamics tire model: Delta F = [Delta Fy, Delta Mz] = F_exp - F_MF62.

Key improvements vs v1:
- PyTorch, float64. The physics gates and the soft saturation are INSIDE the model, so the loss is on the
  deployed (gated) output and the net does not spend capacity where a gate will zero it out.
- Smooth gradients: sqrt(a^2 + eps) and smooth gates replace non-differentiable absolute value / hard clips.
- No anti-symmetry augmentation (which projected the residual onto its odd part against a baseline that is not
  odd due to conicity/ply-steer). The data are the tire as tested; the right wheel is the mirror image at runtime:
  dF_R(alpha, gamma) = -dF_L(-alpha, -gamma).
- Early stopping on contiguous BLOCKS held out from train (no random split of autocorrelated samples).
  The independent validation run is never used for any decision.
- Baseline MF62 is evaluated exactly as the runtime does (model Fx, SVyk in Mz) via evaluate_pure_slip.
- Loss: Huber on residuals scaled by (25 N, 2.5 Nm), load-bin weighted like the calibrator.
- Gate on zero slip is smooth at the origin: 1 - exp(-(alpha^2+gamma^2)/(2 s^2)), s = 6 mrad.
- Soft clip c*tanh(x/c) replaces the hard clip. GDScript implements the identical formulas.
- Exports mf62_calibration_id; the runtime refuses weights that do not match the loaded MF62 calibration.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch import nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from calibrate_mf62 import TIRE_DIR, PROCESSED_DIR, MF62_DIR, VALIDATION_DIR, SCHEMA_VERSION, evaluate_pure_slip  # noqa: E402

NEURAL_DIR = os.path.join(TIRE_DIR, "NeuralCorrection")

SEED = 42
BLOCK_ROWS = 3000          # ~ one sweep; rows inside a block are never split between fit and early-stop sets
INNER_VAL_EVERY = 7        # every 7th block (~14%) is held out for early stopping
BATCH = 1024
MAX_EPOCHS = 400
PATIENCE = 30
LR = 3e-3
WEIGHT_DECAY = 1e-5
LOSS_SCALE = (25.0, 2.5)   # N, Nm
HIDDEN = 16

GATES = {
    "slip_scale_rad": 0.006,
    "load_trans_n": 100.0,
    "alpha_limit_rad": 0.22,
    "alpha_slope_rad": 0.02,
    "fz_limit_n": 1300.0,
    "fz_slope_n": 100.0,
    "clip_fy_n": 200.0,
    "clip_mz_nm": 10.0,
}

torch.set_default_dtype(torch.float64)


def gate_scale(x, G):
    a, fz, g = x[:, 0], x[:, 1], x[:, 2]
    g_slip = 1.0 - torch.exp(-(a * a + g * g) / (2.0 * G["slip_scale_rad"] ** 2))
    t = torch.clamp(fz / G["load_trans_n"], 0.0, 1.0)
    g_load = t * t * (3.0 - 2.0 * t)
    a_abs = torch.sqrt(a * a + 1e-6)
    g_alpha = torch.sigmoid(-(a_abs - G["alpha_limit_rad"]) / G["alpha_slope_rad"])
    g_fz = torch.sigmoid(-(fz - G["fz_limit_n"]) / G["fz_slope_n"])
    return g_slip * g_load * g_alpha * g_fz


class GatedResidual(nn.Module):
    def __init__(self, x_mean, x_std, y_mean, y_std, gates):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                                 nn.Linear(HIDDEN, 2))
        self.register_buffer("xm", torch.tensor(x_mean))
        self.register_buffer("xs", torch.tensor(x_std))
        self.register_buffer("ym", torch.tensor(y_mean))
        self.register_buffer("ys", torch.tensor(y_std))
        self.register_buffer("clip", torch.tensor([gates["clip_fy_n"], gates["clip_mz_nm"]]))
        self.gates = gates

    def forward(self, x_raw):
        y = self.net((x_raw - self.xm) / self.xs) * self.ys + self.ym
        y = y * gate_scale(x_raw, self.gates).unsqueeze(1)
        return self.clip * torch.tanh(y / self.clip)


def load_weights_by_bin(fz):
    bins = np.digitize(fz, [330.0, 550.0, 770.0, 990.0])
    w = 1.0 / np.sqrt(np.maximum(np.bincount(bins, minlength=5)[bins], 1))
    return w / np.mean(w)


def residuals(df, params):
    a, fz, g, p = (df[c].values for c in ("alpha", "f_z", "gamma", "p_kpa"))
    fy_mf, mz_mf, _ = evaluate_pure_slip(params, a, fz, g, p)
    X = np.column_stack([a, fz, g, p])
    Y = np.column_stack([df["f_y"].values - fy_mf, df["m_z"].values - mz_mf])
    return X, Y, fy_mf, mz_mf


def metrics(exp, pred):
    err = pred - exp
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "r2": float(1.0 - np.sum(err ** 2) / max(np.sum((exp - np.mean(exp)) ** 2), 1e-9))}


def train_compound(compound):
    print(f"\n{'=' * 75}\n   RESIDUAL NN: HOOSIER 16.0x7.5-10 {compound}\n{'=' * 75}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    pfile = os.path.join(MF62_DIR, f"hoosier_16x7.5_10_{compound.lower()}_mf62.json")
    with open(pfile) as fp:
        params = json.load(fp)
    if params.get("SCHEMA_VERSION") != SCHEMA_VERSION or "CALIBRATION_ID" not in params:
        sys.exit(f"{pfile}: schema/CALIBRATION_ID mismatch. Re-run calibrate_mf62.py (schema v{SCHEMA_VERSION}).")

    df_tr = pd.read_csv(os.path.join(PROCESSED_DIR, f"{compound}_train.csv"))
    df_va = pd.read_csv(os.path.join(PROCESSED_DIR, f"{compound}_val.csv"))
    X_tr, Y_tr, fy_mf_tr, mz_mf_tr = residuals(df_tr, params)
    X_va, Y_va, fy_mf_va, mz_mf_va = residuals(df_va, params)
    base = {"train": (float(np.sqrt(np.mean(Y_tr[:, 0] ** 2))), float(np.sqrt(np.mean(Y_tr[:, 1] ** 2)))),
            "val": (float(np.sqrt(np.mean(Y_va[:, 0] ** 2))), float(np.sqrt(np.mean(Y_va[:, 1] ** 2))))}
    print(f">> MF62 baseline RMSE  train Fy {base['train'][0]:.1f} N / Mz {base['train'][1]:.2f} Nm | "
          f"val Fy {base['val'][0]:.1f} N / Mz {base['val'][1]:.2f} Nm")

    blk = np.arange(len(X_tr)) // BLOCK_ROWS
    inner_val = (blk % INNER_VAL_EVERY) == (INNER_VAL_EVERY - 1)
    fit = ~inner_val
    print(f">> Fit rows {fit.sum()} | inner early-stop rows {inner_val.sum()} (blocks of {BLOCK_ROWS})")

    x_mean, x_std = X_tr[fit].mean(0), X_tr[fit].std(0)
    x_std[x_std < 1e-6] = 1.0
    y_mean, y_std = Y_tr[fit].mean(0), Y_tr[fit].std(0)
    y_std[y_std < 1e-6] = 1.0

    model = GatedResidual(x_mean, x_std, y_mean, y_std, GATES)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    huber = nn.HuberLoss(reduction="none", delta=1.0)
    scale = torch.tensor(LOSS_SCALE)

    w_all = load_weights_by_bin(X_tr[:, 1])
    Xf, Yf, Wf = (torch.tensor(X_tr[fit]), torch.tensor(Y_tr[fit]), torch.tensor(w_all[fit]))
    Xv, Yv, Wv = (torch.tensor(X_tr[inner_val]), torch.tensor(Y_tr[inner_val]), torch.tensor(w_all[inner_val]))

    def loss_fn(x, y, w):
        return (huber(model(x) / scale, y / scale).sum(1) * w).mean()

    best, best_state, bad, ep = float("inf"), None, 0, 0
    gen = torch.Generator().manual_seed(SEED)
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xf), generator=gen)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(Xf[idx], Yf[idx], Wf[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(Xv, Yv, Wv))
        sched.step(v)
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    converged = bad >= PATIENCE
    print(f">> Stopped at epoch {ep} ({'early stop' if converged else 'MAX_EPOCHS reached: not converged'}); "
          f"best inner-val loss {best:.5f}")

    with torch.no_grad():
        d_tr = model(torch.tensor(X_tr)).numpy()
        d_va = model(torch.tensor(X_va)).numpy()

    out = {}
    for name, df, fy_mf, mz_mf, d, bs in [("train", df_tr, fy_mf_tr, mz_mf_tr, d_tr, base["train"]),
                                          ("validation", df_va, fy_mf_va, mz_mf_va, d_va, base["val"])]:
        mfy = metrics(df["f_y"].values, fy_mf + d[:, 0])
        mmz = metrics(df["m_z"].values, mz_mf + d[:, 1])
        out[name] = {"base_mf62": {"fy_rmse_n": bs[0], "mz_rmse_nm": bs[1]},
                     "hybrid": {"fy": mfy, "mz": mmz},
                     "gain_pct": {"fy": (1 - mfy["rmse"] / bs[0]) * 100.0, "mz": (1 - mmz["rmse"] / bs[1]) * 100.0}}
    sliced = {}
    fzv = X_va[:, 1]
    for lo, hi, nm in [(100, 330, "L1"), (330, 550, "L2"), (550, 770, "L3"), (770, 990, "L4"), (990, 1400, "L5")]:
        m = (fzv >= lo) & (fzv < hi)
        if m.sum() > 100:
            sliced[nm] = {"fy_rmse_base": float(np.sqrt(np.mean(Y_va[m, 0] ** 2))),
                          "fy_rmse_hybrid": float(np.sqrt(np.mean((Y_va[m, 0] - d_va[m, 0]) ** 2))),
                          "mz_rmse_base": float(np.sqrt(np.mean(Y_va[m, 1] ** 2))),
                          "mz_rmse_hybrid": float(np.sqrt(np.mean((Y_va[m, 1] - d_va[m, 1]) ** 2)))}
    out["validation_by_load"] = sliced

    for nm in ("train", "validation"):
        o = out[nm]
        print(f"  [{nm}] Fy {o['base_mf62']['fy_rmse_n']:.1f} -> {o['hybrid']['fy']['rmse']:.1f} N ({o['gain_pct']['fy']:+.1f}%) | "
              f"Mz {o['base_mf62']['mz_rmse_nm']:.2f} -> {o['hybrid']['mz']['rmse']:.2f} Nm ({o['gain_pct']['mz']:+.1f}%)")
    if out["validation"]["gain_pct"]["fy"] < 5.0:
        print("  [WARNING] Fy residual gain on independent validation < 5%: the NN does not generalise for Fy; "
              "consider shipping the Mz correction only or none.")

    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    payload = {
        "architecture": {"compound": compound, "in_dim": 4, "hidden1": HIDDEN, "hidden2": HIDDEN, "out_dim": 2,
                         "inputs": ["alpha", "f_z", "gamma", "p_kpa"], "outputs": ["delta_Fy", "delta_Mz"],
                         "activation": "tanh", "mirror_for_right_side": True},
        "mf62_calibration_id": params["CALIBRATION_ID"],
        "mf62_schema_version": SCHEMA_VERSION,
        "normalization": {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
                          "y_mean": y_mean.tolist(), "y_std": y_std.tolist()},
        "weights": {"w1": sd["net.0.weight"].tolist(), "b1": sd["net.0.bias"].tolist(),
                    "w2": sd["net.2.weight"].tolist(), "b2": sd["net.2.bias"].tolist(),
                    "w3": sd["net.4.weight"].tolist(), "b3": sd["net.4.bias"].tolist()},
        "physics_gates": {k: float(v) for k, v in GATES.items()},
        "training": {"epochs": ep, "converged": bool(converged), "seed": SEED,
                     "validation_gain_pct": out["validation"]["gain_pct"]},
    }
    os.makedirs(NEURAL_DIR, exist_ok=True)
    wpath = os.path.join(NEURAL_DIR, f"neural_residual_weights_{compound.lower()}.json")
    with open(wpath, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f">> Exported {wpath}")
    return {"compound": compound, "epochs": ep, "converged": bool(converged),
            "mf62_calibration_id": params["CALIBRATION_ID"], **out}


def main():
    os.makedirs(NEURAL_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    report = {"model_type": "MF 6.2 (schema v%d) + gated residual MLP 4-16-16-2" % SCHEMA_VERSION}
    for comp in ("R25B", "R20"):
        report[comp] = train_compound(comp)
    # Also write default neural_residual_weights.json (for R25B)
    with open(os.path.join(NEURAL_DIR, "neural_residual_weights_r25b.json")) as f:
        r25b_weights = json.load(f)
    with open(os.path.join(NEURAL_DIR, "neural_residual_weights.json"), "w") as f:
        json.dump(r25b_weights, f, indent=2)
    with open(os.path.join(VALIDATION_DIR, "neural_residual_report.json"), "w") as fp:
        json.dump(report, fp, indent=2)
    print("\n>> Report written.")


if __name__ == "__main__":
    main()
