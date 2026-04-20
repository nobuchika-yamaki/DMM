# mdd_recovery_mechanism_validation.py
# Validate candidate mechanisms for the observed MDD differences:
#   - mean_lag1_autocorr decrease
#   - lowfreq_fraction decrease
#
# Main question:
#   Which candidate mechanism can explain both worsening (control -> MDD direction)
#   and recovery (MDD -> control direction) along the same parameter axis?
#
# Candidate mechanisms:
#   A. recurrent_gain
#   B. slow_drive
#   C. noise
#
# Analysis structure:
#   Stage 1: observational control vs MDD summary
#   Stage 2: fit baseline control models
#   Stage 3: calibrate each mechanism by worsening fit
#   Stage 4: test recovery along the same parameter axis
#
# Usage:
#   python3 mdd_recovery_mechanism_validation.py
#
# Requirement:
#   Place this script in the same folder as mdd_minimal_mapping.py

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import mdd_minimal_mapping as base
except ImportError as exc:
    raise ImportError("mdd_minimal_mapping.py must be in the same folder as this script.") from exc

find_niftis = base.find_niftis
get_pid = base.get_pid
find_tsv = base.find_tsv
load_run = base.load_run
select_top_pairs = base.select_top_pairs
fit_pca = base.fit_pca
project_pca = base.project_pca
run_summary = base.run_summary
aggregate_subject_summaries = base.aggregate_subject_summaries
norm_group = base.norm_group
INPUT_ROOT = base.INPUT_ROOT
SUMMARY_METRICS = base.SUMMARY_METRICS

HOME = Path.home()
OUT_DIR = HOME / "Desktop" / "mdd_recovery_mechanism_results"

SEED = 12345
NPC_OPTIONS = [2, 3, 4]
LQ_OPTIONS = [True, False]

PRIMARY_METRICS = ["mean_lag1_autocorr", "lowfreq_fraction"]
SUPPORT_METRICS = ["mean_speed", "trace_cov"]
TARGET_METRICS = PRIMARY_METRICS + SUPPORT_METRICS

LAMBDA_GRID = np.linspace(0.0, 1.0, 11)

GAIN_ANCHOR_GRID = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
SLOW_ANCHOR_GRID = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
NOISE_ANCHOR_GRID = [1.00, 1.10, 1.25, 1.50, 2.00, 3.00]

RHO_SLOW = 0.90
N_CONTROL_NULL = 100

# penalties are intentionally small: this is a sufficiency test, not winner-takes-all model selection
LAMBDA_GAIN = 0.10
LAMBDA_SLOW = 0.10
LAMBDA_NOISE = 0.10


# ---------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------
def cohens_d(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    pooled = math.sqrt(((len(x) - 1) * sx * sx + (len(y) - 1) * sy * sy) / (len(x) + len(y) - 2))
    if pooled < 1e-12:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / pooled)


def bootstrap_mean_diff(x, y, n_boot=5000, seed=SEED):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        vals[i] = np.mean(yb) - np.mean(xb)
    return float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def permutation_pvalue(x, y, n_perm=5000, seed=SEED):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(y) - np.mean(x))
    pooled = np.concatenate([x, y]).copy()
    nx = len(x)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = abs(np.mean(pooled[nx:]) - np.mean(pooled[:nx]))
        if d >= obs:
            hits += 1
    return float((hits + 1) / (n_perm + 1))


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    qr = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ranked[i] * m / (i + 1))
        qr[i] = prev
    out = np.empty(m)
    out[order] = np.clip(qr, 0, 1)
    q[ok] = out
    return q


def stable_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 1.0
    s = float(np.std(x, ddof=1))
    return s if np.isfinite(s) and s > 1e-6 else 1.0


# ---------------------------------------------------------------------
# loading / preprocessing
# ---------------------------------------------------------------------
def load_all_runs(apply_lq_correction: bool):
    niftis = find_niftis(INPUT_ROOT)
    if not niftis:
        raise FileNotFoundError("No task-rest_bold files found under ~/Downloads")

    pids = {get_pid(p) for p in niftis}
    _, part_df = find_tsv(pids)
    part_df["participant_id"] = part_df["participant_id"].astype(str)
    part_df["group"] = part_df["group"].map(norm_group)

    pid_to_group = dict(zip(part_df["participant_id"], part_df["group"]))

    lq_map = {}
    if "Edinburgh" in part_df.columns and apply_lq_correction:
        for _, row in part_df.iterrows():
            lq = pd.to_numeric(row["Edinburgh"], errors="coerce")
            if pd.notna(lq):
                lq_map[str(row["participant_id"])] = -1.0 if lq < 0 else 1.0

    runs = []
    shape_counts = {}
    for p in niftis:
        pid = get_pid(p)
        group = pid_to_group.get(pid, "unknown")
        lq_sign = lq_map.get(pid, 1.0)
        try:
            D, names, shape = load_run(p, lq_sign=lq_sign)
        except Exception:
            continue
        runs.append({
            "pid": pid,
            "group": group,
            "D": D,
            "names": names,
            "shape": shape,
            "file": str(p),
        })
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    if not runs:
        raise RuntimeError("No analysable runs were loaded.")
    best_shape = max(shape_counts, key=shape_counts.get)
    runs = [r for r in runs if r["shape"] == best_shape]
    return runs, best_shape


def prepare_latent_runs(all_runs: List[Dict], npc: int):
    pair_names = all_runs[0]["names"]
    ctrl_runs = [r for r in all_runs if r["group"] == "control"]
    mdd_runs = [r for r in all_runs if r["group"] == "depr"]
    if len(ctrl_runs) < 5 or len(mdd_runs) < 5:
        raise RuntimeError("Too few control or MDD runs.")

    sel_idx, sel_names = select_top_pairs(ctrl_runs, pair_names)
    X_ctrl = np.vstack([r["D"][:, sel_idx] for r in ctrl_runs])
    mean, sd, basis, pca_info = fit_pca(X_ctrl, npc=npc)

    ctrl_latent = []
    mdd_latent = []
    run_rows = []
    for r in all_runs:
        Z = project_pca(r["D"][:, sel_idx], mean, sd, basis)
        item = {"pid": r["pid"], "group": r["group"], "Z": Z}
        if r["group"] == "control":
            ctrl_latent.append(item)
        elif r["group"] == "depr":
            mdd_latent.append(item)

        row = {"pid": r["pid"], "group": r["group"]}
        row.update(run_summary(Z))
        run_rows.append(row)

    subj_df = aggregate_subject_summaries(run_rows)
    ctrl_subj = subj_df[subj_df["group"] == "control"].copy()
    mdd_subj = subj_df[subj_df["group"] == "depr"].copy()

    ctrl_group = {}
    mdd_group = {}
    ctrl_sd_map = {}
    for m in TARGET_METRICS:
        ctrl_group[m] = float(pd.to_numeric(ctrl_subj[m], errors="coerce").mean())
        mdd_group[m] = float(pd.to_numeric(mdd_subj[m], errors="coerce").mean())
        ctrl_sd_map[m] = stable_sd(pd.to_numeric(ctrl_subj[m], errors="coerce").values)

    return {
        "ctrl_latent": ctrl_latent,
        "mdd_latent": mdd_latent,
        "subject_df": subj_df,
        "ctrl_group": ctrl_group,
        "mdd_group": mdd_group,
        "ctrl_sd_map": ctrl_sd_map,
        "pca_info": pca_info,
        "n_selected_pairs": int(len(sel_idx)),
        "selected_pairs": sel_names,
    }


# ---------------------------------------------------------------------
# Stage 1 observational
# ---------------------------------------------------------------------
def observational_group_diffs(subj_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(SUMMARY_METRICS):
        x = pd.to_numeric(subj_df.loc[subj_df["group"] == "control", m], errors="coerce").dropna().values
        y = pd.to_numeric(subj_df.loc[subj_df["group"] == "depr", m], errors="coerce").dropna().values
        if len(x) < 2 or len(y) < 2:
            continue
        md, lo, hi = bootstrap_mean_diff(x, y, seed=SEED + i)
        pp = permutation_pvalue(x, y, seed=SEED + 10000 + i)
        rows.append({
            "metric": m,
            "n_control": len(x),
            "n_mdd": len(y),
            "control_mean": float(np.mean(x)),
            "control_sd": float(np.std(x, ddof=1)),
            "mdd_mean": float(np.mean(y)),
            "mdd_sd": float(np.std(y, ddof=1)),
            "mean_diff_mdd_minus_control": float(np.mean(y) - np.mean(x)),
            "bootstrap_mean_diff": md,
            "ci_2_5": lo,
            "ci_97_5": hi,
            "cohens_d_mdd_minus_control": cohens_d(y, x),
            "perm_p": pp,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_q"] = bh_fdr(out["perm_p"].values)
        out["abs_d"] = np.abs(out["cohens_d_mdd_minus_control"])
        out = out.sort_values(["fdr_q", "abs_d"], ascending=[True, False]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------
# baseline control models
# ---------------------------------------------------------------------
def fit_ar1_group(latent_runs: List[Dict]) -> Dict[str, np.ndarray]:
    X_prev = []
    X_next = []
    for r in latent_runs:
        Z = np.asarray(r["Z"], dtype=float)
        if len(Z) < 2:
            continue
        X_prev.append(Z[:-1])
        X_next.append(Z[1:])
    X_prev = np.vstack(X_prev)
    X_next = np.vstack(X_next)

    D = X_prev.shape[1]
    phi = np.zeros(D, dtype=float)
    sigma = np.zeros(D, dtype=float)
    for j in range(D):
        x = X_prev[:, j]
        y = X_next[:, j]
        denom = float(np.dot(x, x))
        pj = float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0
        pj = float(np.clip(pj, -0.995, 0.995))
        resid = y - pj * x
        sj = float(np.std(resid, ddof=1)) if len(resid) > 1 else 1e-3
        phi[j] = pj
        sigma[j] = max(sj, 1e-3)
    return {"phi": phi, "sigma": sigma}


def fit_slowdrive_group(latent_runs: List[Dict], rho: float = RHO_SLOW) -> Dict[str, np.ndarray]:
    a_all = []
    b_all = []
    sigma_all = []
    D = None
    for r in latent_runs:
        Z = np.asarray(r["Z"], dtype=float)
        D = Z.shape[1]
        break
    if D is None:
        raise RuntimeError("No latent runs found for slow-drive fit.")

    for j in range(D):
        X_list = []
        y_list = []
        for r in latent_runs:
            z = np.asarray(r["Z"][:, j], dtype=float)
            if len(z) < 3:
                continue
            s = np.zeros_like(z)
            s[0] = z[0]
            for t in range(1, len(z)):
                s[t] = rho * s[t - 1] + (1.0 - rho) * z[t - 1]
            X = np.column_stack([z[:-1], s[:-1], np.ones(len(z) - 1)])
            y = z[1:]
            X_list.append(X)
            y_list.append(y)
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = X @ beta
        resid = y - pred
        a_all.append(float(beta[0]))
        b_all.append(float(beta[1]))
        sigma_all.append(max(float(np.std(resid, ddof=1)), 1e-3))

    return {
        "a": np.asarray(a_all, dtype=float),
        "b": np.asarray(b_all, dtype=float),
        "sigma": np.asarray(sigma_all, dtype=float),
        "rho": float(rho),
    }


# ---------------------------------------------------------------------
# simulation helpers
# ---------------------------------------------------------------------
def simulate_gain_model(length: int, z0: np.ndarray, ctrl_ar1: Dict[str, np.ndarray], gain_scale: float, rng: np.random.Generator) -> np.ndarray:
    phi = np.asarray(ctrl_ar1["phi"], dtype=float) * float(gain_scale)
    sigma = np.asarray(ctrl_ar1["sigma"], dtype=float)
    D = len(phi)
    Z = np.zeros((length, D), dtype=float)
    Z[0] = np.asarray(z0, dtype=float)
    for t in range(1, length):
        eps = rng.normal(0.0, sigma, size=D)
        Z[t] = phi * Z[t - 1] + eps
    return np.clip(Z, -10.0, 10.0)


def simulate_noise_model(length: int, z0: np.ndarray, ctrl_ar1: Dict[str, np.ndarray], noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    phi = np.asarray(ctrl_ar1["phi"], dtype=float)
    sigma = np.asarray(ctrl_ar1["sigma"], dtype=float) * float(noise_scale)
    D = len(phi)
    Z = np.zeros((length, D), dtype=float)
    Z[0] = np.asarray(z0, dtype=float)
    for t in range(1, length):
        eps = rng.normal(0.0, sigma, size=D)
        Z[t] = phi * Z[t - 1] + eps
    return np.clip(Z, -10.0, 10.0)


def simulate_slowdrive_model(length: int, z0: np.ndarray, ctrl_slow: Dict[str, np.ndarray], slow_scale: float, rng: np.random.Generator) -> np.ndarray:
    a = np.asarray(ctrl_slow["a"], dtype=float)
    b = np.asarray(ctrl_slow["b"], dtype=float) * float(slow_scale)
    sigma = np.asarray(ctrl_slow["sigma"], dtype=float)
    rho = float(ctrl_slow["rho"])
    D = len(a)

    Z = np.zeros((length, D), dtype=float)
    S = np.zeros((length, D), dtype=float)
    Z[0] = np.asarray(z0, dtype=float)
    S[0] = np.asarray(z0, dtype=float)

    for t in range(1, length):
        eps = rng.normal(0.0, sigma, size=D)
        Z[t] = a * Z[t - 1] + b * S[t - 1] + eps
        S[t] = rho * S[t - 1] + (1.0 - rho) * Z[t - 1]
    return np.clip(Z, -10.0, 10.0)


def summarize_group(latent_runs: List[Dict]) -> Dict[str, float]:
    rows = []
    for r in latent_runs:
        row = {"pid": r["pid"], "group": r["group"]}
        row.update(run_summary(r["Z"]))
        rows.append(row)
    subj_df = aggregate_subject_summaries(rows)
    out = {}
    for m in TARGET_METRICS:
        out[m] = float(pd.to_numeric(subj_df[m], errors="coerce").mean())
    return out


def primary_loss(sim_summary: Dict[str, float], target_summary: Dict[str, float], ctrl_sd_map: Dict[str, float]) -> float:
    vals = []
    for m in PRIMARY_METRICS:
        sv = sim_summary.get(m, np.nan)
        tv = target_summary.get(m, np.nan)
        vals.append(abs(sv - tv) / ctrl_sd_map[m] if np.isfinite(sv) and np.isfinite(tv) else np.nan)
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def support_loss(sim_summary: Dict[str, float], target_summary: Dict[str, float], ctrl_sd_map: Dict[str, float]) -> float:
    vals = []
    for m in SUPPORT_METRICS:
        sv = sim_summary.get(m, np.nan)
        tv = target_summary.get(m, np.nan)
        vals.append(abs(sv - tv) / ctrl_sd_map[m] if np.isfinite(sv) and np.isfinite(tv) else np.nan)
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def recovery_index(sim_summary: Dict[str, float], ctrl_summary: Dict[str, float], mdd_summary: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for m in PRIMARY_METRICS:
        num = sim_summary[m] - mdd_summary[m]
        den = ctrl_summary[m] - mdd_summary[m]
        out[f"RI_{m}"] = float(num / den) if np.isfinite(den) and abs(den) > 1e-12 else np.nan
    return out


# ---------------------------------------------------------------------
# mechanism paths
# ---------------------------------------------------------------------
def simulate_worsening_runs(prep: Dict, mechanism: str, anchor: float, lam: float, ctrl_ar1: Dict[str, np.ndarray], ctrl_slow: Dict[str, np.ndarray], rng: np.random.Generator) -> List[Dict]:
    out = []
    ctrl_pool = prep["ctrl_latent"]
    n_ctrl = len(ctrl_pool)

    if mechanism == "recurrent_gain":
        param = 1.0 + lam * (anchor - 1.0)
    elif mechanism == "slow_drive":
        param = 1.0 + lam * (anchor - 1.0)
    elif mechanism == "noise":
        param = 1.0 + lam * (anchor - 1.0)
    else:
        raise ValueError(mechanism)

    for tmpl in prep["mdd_latent"]:
        src = ctrl_pool[rng.integers(0, n_ctrl)]
        z0 = src["Z"][0].copy()
        T = len(tmpl["Z"])
        if mechanism == "recurrent_gain":
            Z = simulate_gain_model(T, z0, ctrl_ar1, gain_scale=param, rng=rng)
        elif mechanism == "slow_drive":
            Z = simulate_slowdrive_model(T, z0, ctrl_slow, slow_scale=param, rng=rng)
        else:
            Z = simulate_noise_model(T, z0, ctrl_ar1, noise_scale=param, rng=rng)
        out.append({"pid": tmpl["pid"], "group": "depr", "Z": Z})
    return out


def simulate_recovery_runs(prep: Dict, mechanism: str, anchor: float, lam: float, ctrl_ar1: Dict[str, np.ndarray], ctrl_slow: Dict[str, np.ndarray], rng: np.random.Generator) -> List[Dict]:
    out = []

    if mechanism in ("recurrent_gain", "slow_drive"):
        # anchor < 1, recovery goes from anchor back to 1
        param = anchor + lam * (1.0 - anchor)
    elif mechanism == "noise":
        # anchor > 1, recovery goes from anchor back to 1
        param = anchor + lam * (1.0 - anchor)
    else:
        raise ValueError(mechanism)

    for tmpl in prep["mdd_latent"]:
        z0 = tmpl["Z"][0].copy()
        T = len(tmpl["Z"])
        if mechanism == "recurrent_gain":
            Z = simulate_gain_model(T, z0, ctrl_ar1, gain_scale=param, rng=rng)
        elif mechanism == "slow_drive":
            Z = simulate_slowdrive_model(T, z0, ctrl_slow, slow_scale=param, rng=rng)
        else:
            Z = simulate_noise_model(T, z0, ctrl_ar1, noise_scale=param, rng=rng)
        out.append({"pid": tmpl["pid"], "group": "depr", "Z": Z})
    return out


# ---------------------------------------------------------------------
# calibration and validation
# ---------------------------------------------------------------------
def calibrate_anchor(prep: Dict, mechanism: str, ctrl_ar1: Dict[str, np.ndarray], ctrl_slow: Dict[str, np.ndarray]) -> Tuple[float, pd.DataFrame]:
    if mechanism == "recurrent_gain":
        grid = GAIN_ANCHOR_GRID
    elif mechanism == "slow_drive":
        grid = SLOW_ANCHOR_GRID
    elif mechanism == "noise":
        grid = NOISE_ANCHOR_GRID
    else:
        raise ValueError(mechanism)

    rows = []
    for i, anchor in enumerate(grid):
        rng = np.random.default_rng(SEED + 1000 + i)
        sim_runs = simulate_worsening_runs(prep, mechanism, anchor, 1.0, ctrl_ar1, ctrl_slow, rng)
        sim_summary = summarize_group(sim_runs)
        loss_w = primary_loss(sim_summary, prep["mdd_group"], prep["ctrl_sd_map"])
        loss_s = support_loss(sim_summary, prep["mdd_group"], prep["ctrl_sd_map"])

        if mechanism == "recurrent_gain":
            pen = LAMBDA_GAIN * (1.0 - anchor) ** 2
        elif mechanism == "slow_drive":
            pen = LAMBDA_SLOW * (1.0 - anchor) ** 2
        else:
            pen = LAMBDA_NOISE * (anchor - 1.0) ** 2

        total = loss_w + 0.25 * loss_s + pen
        rows.append({
            "mechanism": mechanism,
            "anchor_value": float(anchor),
            "worsening_primary_loss": float(loss_w),
            "worsening_support_loss": float(loss_s),
            "penalty": float(pen),
            "total_score": float(total),
        })
    df = pd.DataFrame(rows).sort_values(["total_score", "worsening_primary_loss", "anchor_value"]).reset_index(drop=True)
    return float(df.iloc[0]["anchor_value"]), df


def control_null_baseline_frequency(prep: Dict, mechanism: str, anchor: float, ctrl_ar1: Dict[str, np.ndarray], ctrl_slow: Dict[str, np.ndarray], n_reps: int = N_CONTROL_NULL) -> float:
    ctrl_runs = prep["ctrl_latent"]
    n_ctrl = len(ctrl_runs)
    wins = 0
    for rep in range(n_reps):
        rng = np.random.default_rng(SEED + 500000 + rep)

        # create control-like null target from baseline control dynamics
        null_runs = []
        for tmpl in ctrl_runs:
            src = ctrl_runs[rng.integers(0, n_ctrl)]
            z0 = src["Z"][0].copy()
            T = len(tmpl["Z"])
            Z = simulate_gain_model(T, z0, ctrl_ar1, gain_scale=1.0, rng=rng)
            null_runs.append({"pid": tmpl["pid"], "group": "control", "Z": Z})
        null_summary = summarize_group(null_runs)

        # baseline loss
        baseline_loss = primary_loss(prep["ctrl_group"], null_summary, prep["ctrl_sd_map"])

        # anchor loss under candidate mechanism
        if mechanism == "recurrent_gain":
            sim_runs = simulate_worsening_runs(prep, mechanism, anchor, 1.0, ctrl_ar1, ctrl_slow, rng)
        elif mechanism == "slow_drive":
            sim_runs = simulate_worsening_runs(prep, mechanism, anchor, 1.0, ctrl_ar1, ctrl_slow, rng)
        else:
            sim_runs = simulate_worsening_runs(prep, mechanism, anchor, 1.0, ctrl_ar1, ctrl_slow, rng)
        sim_summary = summarize_group(sim_runs)
        anchor_loss = primary_loss(sim_summary, null_summary, prep["ctrl_sd_map"])

        if baseline_loss <= anchor_loss:
            wins += 1
    return float(wins / n_reps)


# ---------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------
def save_primary_curves(df: pd.DataFrame, out_png: Path, title: str):
    if len(df) == 0:
        return
    mechs = list(df["mechanism"].dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for mech in mechs:
        sub = df[df["mechanism"] == mech]
        axes[0, 0].plot(sub["lambda"], sub["sim_mean_lag1_autocorr"], marker="o", label=mech)
        axes[0, 1].plot(sub["lambda"], sub["sim_lowfreq_fraction"], marker="o", label=mech)
        axes[1, 0].plot(sub["lambda"], sub["primary_loss"], marker="o", label=mech)
        axes[1, 1].plot(sub["lambda"], sub["RI_mean_lag1_autocorr"], marker="o", label=f"{mech}: lag1")
        axes[1, 1].plot(sub["lambda"], sub["RI_lowfreq_fraction"], marker="x", linestyle="--", label=f"{mech}: lowfreq")

    axes[0, 0].set_title("mean_lag1_autocorr")
    axes[0, 1].set_title("lowfreq_fraction")
    axes[1, 0].set_title("primary_loss")
    axes[1, 1].set_title("Recovery Index")

    for ax in axes.ravel():
        ax.set_xlabel("lambda")
        ax.legend(fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# main per-config runner
# ---------------------------------------------------------------------
def run_one_config(all_runs: List[Dict], npc: int, lq_on: bool, best_shape: str):
    prep = prepare_latent_runs(all_runs, npc=npc)
    obs_diff = observational_group_diffs(prep["subject_df"])
    ctrl_ar1 = fit_ar1_group(prep["ctrl_latent"])
    ctrl_slow = fit_slowdrive_group(prep["ctrl_latent"], rho=RHO_SLOW)

    mechanism_rows = []
    worsen_rows = []
    recover_rows = []
    sim_rows = []
    ri_rows = []
    null_rows = []
    calibration_rows = []

    for mechanism in ["recurrent_gain", "slow_drive", "noise"]:
        anchor, cal_df = calibrate_anchor(prep, mechanism, ctrl_ar1, ctrl_slow)
        cal_df["config_id"] = f"npc{npc}_{'LQon' if lq_on else 'LQoff'}"
        calibration_rows.append(cal_df)

        # worsening path
        for i, lam in enumerate(LAMBDA_GRID):
            rng = np.random.default_rng(SEED + 100000 + i + (0 if mechanism == "recurrent_gain" else 1000 if mechanism == "slow_drive" else 2000))
            sim_runs = simulate_worsening_runs(prep, mechanism, anchor, float(lam), ctrl_ar1, ctrl_slow, rng)
            sim_summary = summarize_group(sim_runs)
            p_loss = primary_loss(sim_summary, prep["mdd_group"], prep["ctrl_sd_map"])
            s_loss = support_loss(sim_summary, prep["mdd_group"], prep["ctrl_sd_map"])
            worsen_rows.append({
                "config_id": f"npc{npc}_{'LQon' if lq_on else 'LQoff'}",
                "npc": npc,
                "lq_correction": bool(lq_on),
                "mechanism": mechanism,
                "anchor_value": float(anchor),
                "lambda": float(lam),
                "primary_loss": float(p_loss),
                "support_loss": float(s_loss),
                "sim_mean_lag1_autocorr": sim_summary["mean_lag1_autocorr"],
                "sim_lowfreq_fraction": sim_summary["lowfreq_fraction"],
                "sim_mean_speed": sim_summary["mean_speed"],
                "sim_trace_cov": sim_summary["trace_cov"],
            })
            sim_rows.append({
                "config_id": f"npc{npc}_{'LQon' if lq_on else 'LQoff'}",
                "stage": "worsening",
                "mechanism": mechanism,
                "lambda": float(lam),
                **sim_summary,
            })

        # recovery path
        for i, lam in enumerate(LAMBDA_GRID):
            rng = np.random.default_rng(SEED + 200000 + i + (0 if mechanism == "recurrent_gain" else 1000 if mechanism == "slow_drive" else 2000))
            sim_runs = simulate_recovery_runs(prep, mechanism, anchor, float(lam), ctrl_ar1, ctrl_slow, rng)
            sim_summary = summarize_group(sim_runs)
            p_loss = primary_loss(sim_summary, prep["ctrl_group"], prep["ctrl_sd_map"])
            s_loss = support_loss(sim_summary, prep["ctrl_group"], prep["ctrl_sd_map"])
            ri = recovery_index(sim_summary, prep["ctrl_group"], prep["mdd_group"])
            recover_rows.append({
                "config_id": f"npc{npc}_{'LQon' if lq_on else 'LQoff'}",
                "npc": npc,
                "lq_correction": bool(lq_on),
                "mechanism": mechanism,
                "anchor_value": float(anchor),
                "lambda": float(lam),
                "primary_loss": float(p_loss),
                "support_loss": float(s_loss),
                "sim_mean_lag1_autocorr": sim_summary["mean_lag1_autocorr"],
                "sim_lowfreq_fraction": sim_summary["lowfreq_fraction"],
                "sim_mean_speed": sim_summary["mean_speed"],
                "sim_trace_cov": sim_summary["trace_cov"],
                **ri,
            })
            ri_rows.append({
                "config_id": f"npc{npc}_{'LQon' if lq_on else 'LQoff'}",
                "mechanism": mechanism,
                "lambda": float(lam),
                **ri,
            })
            sim_rows.append({
                "config_id": f"npc{npc}_{'LQon' if lq_on else 'LQoff'}",
                "stage": "recovery",
                "mechanism": mechanism,
                "lambda": float(lam),
                **sim_summary,
            })

        # per-mechanism summary
        worsen_df = pd.DataFrame([r for r in worsen_rows if r["mechanism"] == mechanism and r["config_id"] == f"npc{npc}_{'LQon' if lq_on else 'LQoff'}"])
        recover_df = pd.DataFrame([r for r in recover_rows if r["mechanism"] == mechanism and r["config_id"] == f"npc{npc}_{'LQon' if lq_on else 'LQoff'}"])

        best_w = worsen_df.sort_values(["primary_loss", "support_loss", "lambda"]).iloc[0]
        best_r = recover_df.sort_values(["primary_loss", "support_loss", "lambda"]).iloc[0]
        null_freq = control_null_baseline_frequency(prep, mechanism, anchor, ctrl_ar1, ctrl_slow, n_reps=N_CONTROL_NULL)
        null_rows.append({
            "config_id": f"npc{npc}_{'LQon' if lq_on else 'LQoff'}",
            "mechanism": mechanism,
            "baseline_preferred_frequency": float(null_freq),
        })

        mechanism_rows.append({
            "config_id": f"npc{npc}_{'LQon' if lq_on else 'LQoff'}",
            "npc": npc,
            "lq_correction": bool(lq_on),
            "best_shape": best_shape,
            "cum_evr": float(prep["pca_info"]["cum_evr"]),
            "n_control_subjects": int((prep["subject_df"]["group"] == "control").sum()),
            "n_mdd_subjects": int((prep["subject_df"]["group"] == "depr").sum()),
            "n_selected_pairs": prep["n_selected_pairs"],
            "mechanism": mechanism,
            "anchor_value": float(anchor),
            "best_worsening_lambda": float(best_w["lambda"]),
            "best_worsening_primary_loss": float(best_w["primary_loss"]),
            "best_recovery_lambda": float(best_r["lambda"]),
            "best_recovery_primary_loss": float(best_r["primary_loss"]),
            "best_recovery_RI_lag1": float(best_r["RI_mean_lag1_autocorr"]),
            "best_recovery_RI_lowfreq": float(best_r["RI_lowfreq_fraction"]),
            "baseline_preferred_frequency": float(null_freq),
            "obs_ctrl_mean_lag1_autocorr": float(prep["ctrl_group"]["mean_lag1_autocorr"]),
            "obs_mdd_mean_lag1_autocorr": float(prep["mdd_group"]["mean_lag1_autocorr"]),
            "obs_ctrl_lowfreq_fraction": float(prep["ctrl_group"]["lowfreq_fraction"]),
            "obs_mdd_lowfreq_fraction": float(prep["mdd_group"]["lowfreq_fraction"]),
        })

    return {
        "obs_diff": obs_diff,
        "mechanism_summary": pd.DataFrame(mechanism_rows),
        "worsen_df": pd.DataFrame(worsen_rows),
        "recover_df": pd.DataFrame(recover_rows),
        "sim_df": pd.DataFrame(sim_rows),
        "ri_df": pd.DataFrame(ri_rows),
        "null_df": pd.DataFrame(null_rows),
        "calibration_df": pd.concat(calibration_rows, ignore_index=True),
        "selected_pairs": prep["selected_pairs"],
        "subject_df": prep["subject_df"],
    }


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_mechanism_rows = []
    all_worsen = []
    all_recover = []
    all_null = []
    all_obs = []
    all_calibration = []

    for lq_on in LQ_OPTIONS:
        all_runs, best_shape = load_all_runs(apply_lq_correction=lq_on)
        for npc in NPC_OPTIONS:
            config_id = f"npc{npc}_{'LQon' if lq_on else 'LQoff'}"
            cfg_dir = OUT_DIR / config_id
            cfg_dir.mkdir(parents=True, exist_ok=True)

            res = run_one_config(all_runs, npc=npc, lq_on=lq_on, best_shape=best_shape)

            res["subject_df"].to_csv(cfg_dir / "subject_summaries.csv", index=False)
            res["obs_diff"].to_csv(cfg_dir / "group_differences.csv", index=False)
            res["mechanism_summary"].to_csv(cfg_dir / "mechanism_summary.csv", index=False)
            res["worsen_df"].to_csv(cfg_dir / "loss_worsening.csv", index=False)
            res["recover_df"].to_csv(cfg_dir / "loss_recovery.csv", index=False)
            res["ri_df"].to_csv(cfg_dir / "ri_by_lambda.csv", index=False)
            res["sim_df"].to_csv(cfg_dir / "simulated_group_means.csv", index=False)
            res["null_df"].to_csv(cfg_dir / "control_null_baseline_preference.csv", index=False)
            res["calibration_df"].to_csv(cfg_dir / "calibration_grid.csv", index=False)
            pd.DataFrame({"selected_pair_name": res["selected_pairs"]}).to_csv(cfg_dir / "selected_pairs.csv", index=False)

            curve_df = res["recover_df"].copy()
            save_primary_curves(curve_df, cfg_dir / "curves_primary.png", config_id)

            all_mechanism_rows.append(res["mechanism_summary"])
            all_worsen.append(res["worsen_df"])
            all_recover.append(res["recover_df"])
            all_null.append(res["null_df"])
            tmp_obs = res["obs_diff"].copy()
            tmp_obs["config_id"] = config_id
            all_obs.append(tmp_obs)
            all_calibration.append(res["calibration_df"])

    mechanism_df = pd.concat(all_mechanism_rows, ignore_index=True)
    worsen_df = pd.concat(all_worsen, ignore_index=True)
    recover_df = pd.concat(all_recover, ignore_index=True)
    null_df = pd.concat(all_null, ignore_index=True)
    obs_df = pd.concat(all_obs, ignore_index=True)
    calibration_df = pd.concat(all_calibration, ignore_index=True)

    # comparison: choose best mechanism per config by worsening + recovery combined
    comp_rows = []
    for config_id, sub in mechanism_df.groupby("config_id"):
        sub = sub.copy()
        sub["combined_score"] = sub["best_worsening_primary_loss"] + sub["best_recovery_primary_loss"]
        best = sub.sort_values(
            ["combined_score", "baseline_preferred_frequency", "best_recovery_RI_lag1", "best_recovery_RI_lowfreq"],
            ascending=[True, False, False, False]
        ).iloc[0]
        comp_rows.append({
            "config_id": config_id,
            "best_mechanism": best["mechanism"],
            "combined_score": float(best["combined_score"]),
            "best_worsening_primary_loss": float(best["best_worsening_primary_loss"]),
            "best_recovery_primary_loss": float(best["best_recovery_primary_loss"]),
            "best_recovery_RI_lag1": float(best["best_recovery_RI_lag1"]),
            "best_recovery_RI_lowfreq": float(best["best_recovery_RI_lowfreq"]),
            "baseline_preferred_frequency": float(best["baseline_preferred_frequency"]),
        })
    comparison_df = pd.DataFrame(comp_rows)

    # sensitivity summary
    sens_rows = []
    for mechanism, sub in mechanism_df.groupby("mechanism"):
        sens_rows.append({
            "mechanism": mechanism,
            "n_configs": int(len(sub)),
            "median_best_worsening_primary_loss": float(np.median(sub["best_worsening_primary_loss"])),
            "median_best_recovery_primary_loss": float(np.median(sub["best_recovery_primary_loss"])),
            "median_best_recovery_RI_lag1": float(np.median(sub["best_recovery_RI_lag1"])),
            "median_best_recovery_RI_lowfreq": float(np.median(sub["best_recovery_RI_lowfreq"])),
            "median_baseline_preferred_frequency": float(np.median(sub["baseline_preferred_frequency"])),
        })
    sensitivity_df = pd.DataFrame(sens_rows).sort_values(
        ["median_best_worsening_primary_loss", "median_best_recovery_primary_loss"],
        ascending=[True, True]
    ).reset_index(drop=True)

    mechanism_df.to_csv(OUT_DIR / "mechanism_config_summary.csv", index=False)
    worsen_df.to_csv(OUT_DIR / "worsening_curve_primary.csv", index=False)
    recover_df.to_csv(OUT_DIR / "recovery_curve_primary.csv", index=False)
    comparison_df.to_csv(OUT_DIR / "mechanism_comparison.csv", index=False)
    sensitivity_df.to_csv(OUT_DIR / "sensitivity_summary.csv", index=False)
    null_df.to_csv(OUT_DIR / "control_null_baseline_preference_all_configs.csv", index=False)
    obs_df.to_csv(OUT_DIR / "observational_group_differences_all_configs.csv", index=False)
    calibration_df.to_csv(OUT_DIR / "calibration_grid_all_configs.csv", index=False)

    lines = []
    lines.append("mdd_recovery_mechanism_validation summary")
    lines.append("")
    lines.append("Primary question:")
    lines.append("Which candidate mechanism explains both worsening (control -> MDD) and recovery (MDD -> control)")
    lines.append("for mean_lag1_autocorr and lowfreq_fraction along the same parameter axis?")
    lines.append("")
    lines.append("Candidate mechanisms:")
    lines.append("- recurrent_gain")
    lines.append("- slow_drive")
    lines.append("- noise")
    lines.append("")
    lines.append("Best mechanism by config:")
    lines.append(comparison_df.to_string(index=False))
    lines.append("")
    lines.append("Sensitivity summary:")
    lines.append(sensitivity_df.to_string(index=False))
    lines.append("")
    lines.append("Interpretation rule:")
    lines.append("A mechanism is favored when it shows low worsening loss, low recovery loss, positive RI on both primary metrics,")
    lines.append("and high control-null baseline preference.")
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()

