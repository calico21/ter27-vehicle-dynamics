#!/usr/bin/env python3
"""
Tire Model Validation Framework — Metrics Core
Calculates statistical, physical, and domain-sliced metrics for model evaluation.
"""

import numpy as np


def compute_metrics(y_true, y_pred):
    """
    Computes standard statistical goodness-of-fit and residual metrics.
    
    Args:
        y_true: np.ndarray of ground truth experimental values
        y_pred: np.ndarray of model predictions
        
    Returns:
        dict containing rmse, mae, max_err, bias, std, nrmse_pct, r2, q90, q95, q99, n
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {
            "rmse": 0.0, "mae": 0.0, "max_err": 0.0, "bias": 0.0, "std": 0.0,
            "nrmse_pct": 0.0, "r2": 0.0, "q90": 0.0, "q95": 0.0, "q99": 0.0, "n": 0
        }

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    max_err = float(np.max(np.abs(err)))
    bias = float(np.mean(err))
    std = float(np.std(err))

    y_range = max(float(np.max(y_true) - np.min(y_true)), 1e-4)
    nrmse = float((rmse / y_range) * 100.0)

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(err ** 2))
    if ss_tot < 1e-6:
        r2 = 1.0 if ss_res < 1e-6 else 0.0
    else:
        r2 = float(max(-10.0, 1.0 - ss_res / ss_tot))

    abs_err = np.abs(err)
    q90 = float(np.percentile(abs_err, 90))
    q95 = float(np.percentile(abs_err, 95))
    q99 = float(np.percentile(abs_err, 99))

    return {
        "rmse": rmse,
        "mae": mae,
        "max_err": max_err,
        "bias": bias,
        "std": std,
        "nrmse_pct": nrmse,
        "r2": r2,
        "q90": q90,
        "q95": q95,
        "q99": q99,
        "n": int(n)
    }


def decompose_error(y_true, y_pred):
    """
    Decomposes total mean squared error (MSE) into systematic bias and variance:
    MSE = Bias^2 + Variance.
    Also separates systematic condition trend from stochastic noise.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"mse": 0.0, "bias_sq": 0.0, "variance": 0.0, "pct_bias": 0.0, "pct_var": 0.0}

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    bias = float(np.mean(err))
    bias_sq = float(bias ** 2)
    var = float(np.var(err))

    pct_bias = float((bias_sq / max(mse, 1e-6)) * 100.0)
    pct_var = float((var / max(mse, 1e-6)) * 100.0)

    return {
        "mse": mse,
        "bias": bias,
        "bias_sq": bias_sq,
        "variance": var,
        "pct_bias": pct_bias,
        "pct_variance": pct_var
    }


def bootstrap_confidence_interval(y_true, y_pred, n_boot=200, ci=0.95, seed=42):
    """
    Calculates bootstrap confidence intervals for RMSE and R^2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n < 20:
        m = compute_metrics(y_true, y_pred)
        return {
            "rmse_mean": m["rmse"], "rmse_ci_lo": m["rmse"], "rmse_ci_hi": m["rmse"],
            "r2_mean": m["r2"], "r2_ci_lo": m["r2"], "r2_ci_hi": m["r2"]
        }

    rng = np.random.RandomState(seed)
    rmse_list = []
    r2_list = []
    alpha_ci = (1.0 - ci) / 2.0

    for _ in range(n_boot):
        indices = rng.randint(0, n, size=n)
        sub_true = y_true[indices]
        sub_pred = y_pred[indices]
        err = sub_pred - sub_true
        rmse_b = np.sqrt(np.mean(err ** 2))
        ss_tot = np.sum((sub_true - np.mean(sub_true)) ** 2)
        ss_res = np.sum(err ** 2)
        r2_b = 1.0 - ss_res / max(ss_tot, 1e-6)
        rmse_list.append(rmse_b)
        r2_list.append(r2_b)

    return {
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_ci_lo": float(np.percentile(rmse_list, alpha_ci * 100.0)),
        "rmse_ci_hi": float(np.percentile(rmse_list, (1.0 - alpha_ci) * 100.0)),
        "r2_mean": float(np.mean(r2_list)),
        "r2_ci_lo": float(np.percentile(r2_list, alpha_ci * 100.0)),
        "r2_ci_hi": float(np.percentile(r2_list, (1.0 - alpha_ci) * 100.0))
    }
