# mdd_observation_ar1.py
# Two-stage analysis:
#   Stage 1: simple observational control vs MDD comparison
#   Stage 2: AR(1) sufficiency test for the two primary metrics that showed
#            the most consistent directional difference
#
# Primary question:
#   What basic dynamical features most reproducibly differ in MDD?
#
# Secondary question:
#   Can a simple AR(1) model reproduce the observed direction of those differences?
#
# Usage:
#   python3 mdd_observation_ar1.py
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

# reuse existing preprocessing
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
OUT_DIR = HOME / "Desktop" / "mdd_observation_ar1_results"

SEED = 12345
NPC_OPTIONS = [2, 3, 4]
LQ_OPTIONS = [True, False]

PRIMARY_METRICS = ["mean_lag1_autocorr", "lowfreq_fraction"]
SUPPORT_METRICS = ["mean_speed", "trace_cov"]

N_BOOT = 5000
N_PERM = 5000
N_PPC = 200
N_CONTROL_NULL = 200

PHI_SCALE_GRID = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70]
NOISE_SCALE_GRID = [1.00, 1.10, 1.25, 1.50, 2.00]

LAMBDA_PHI = 0.20
LAMBDA_NOISE = 0.20


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


def bootstrap_mean_diff(x, y, n_boot=N_BOOT, seed=SEED):
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
        vals[i] = np.mean(yb) - np.mean(xb)  # MDD - control
    return float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def permutation_pvalue(x, y, n_perm=N_PERM, seed=SEED):
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


def stable_sd(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 1.0
    s = float(np.std(x, ddof=1))
    return s if np.isfinite(s) and s > 1e-6 else 1.0


# ---------------------------------------------------------------------
# data loading / preprocessing
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
    depr_runs = [r for r in all_runs if r["group"] == "depr"]

    if len(ctrl_runs) < 5 or len(depr_runs) < 5:
        raise RuntimeError("Too few control or MDD runs.")

    sel_idx, sel_names = select_top_pairs(ctrl_runs, pair_names)
    X_ctrl = np.vstack([r["D"][:, sel_idx] for r in ctrl_runs])
    mean, sd, basis, pca_info = fit_pca(X_ctrl, npc=npc)

    ctrl_latent = []
    depr_latent = []
    run_rows = []
    for r in all_runs:
        Z = project_pca(r["D"][:, sel_idx], mean, sd, basis)
        item = {"pid": r["pid"], "group": r["group"], "Z": Z}
        if r["group"] == "control":
            ctrl_latent.append(item)
        elif r["group"] == "depr":
            depr_latent.append(item)

        row = {"pid": r["pid"], "group": r["group"]}
        row.update(run_summary(Z))
        run_rows.append(row)

    subj_df = aggregate_subject_summaries(run_rows)
    ctrl_subj = subj_df[subj_df["group"] == "control"].copy()
    mdd_subj = subj_df[subj_df["group"] == "depr"].copy()

    ctrl_sd_map = {}
    for m in PRIMARY_METRICS + SUPPORT_METRICS:
        ctrl_sd_map[m] = stable_sd(pd.to_numeric(ctrl_subj[m], errors="coerce").values)

    ctrl_group = {}
    mdd_group = {}
    for m in PRIMARY_METRICS + SUPPORT_METRICS:
        ctrl_group[m] = float(pd.to_numeric(ctrl_subj[m], errors="coerce").mean())
        mdd_group[m] = float(pd.to_numeric(mdd_subj[m], errors="coerce").mean())

    return {
        "ctrl_latent": ctrl_latent,
        "mdd_latent": depr_latent,
        "subject_df": subj_df,
        "ctrl_group": ctrl_group,
        "mdd_group": mdd_group,
        "ctrl_sd_map": ctrl_sd_map,
        "pca_info": pca_info,
        "n_selected_pairs": int(len(sel_idx)),
        "selected_pairs": sel_names,
    }


# ---------------------------------------------------------------------
# Stage 1: observational comparison
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
# Stage 2: AR(1) sufficiency test
# ---------------------------------------------------------------------
def fit_ar1_model(ctrl_latent: List[Dict]) -> Dict[str, np.ndarray]:
    X_prev = []
    X_next = []
    for r in ctrl_latent:
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


def simulate_ar1(length: int, start_z: np.ndarray, ar1: Dict[str, np.ndarray], phi_scale: float, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    phi = np.asarray(ar1["phi"], dtype=float) * float(phi_scale)
    sigma = np.asarray(ar1["sigma"], dtype=float) * float(noise_scale)
    D = len(phi)
    Z = np.zeros((length, D), dtype=float)
    Z[0] = np.asarray(start_z, dtype=float)
    for t in range(1, length):
        eps = rng.normal(0.0, sigma, size=D)
        Z[t] = phi * Z[t - 1] + eps
    return np.clip(Z, -10.0, 10.0)


def aggregate_group_summary(latent_runs: List[Dict]) -> Dict[str, float]:
    rows = []
    for r in latent_runs:
        row = {"pid": r["pid"], "group": r["group"]}
        row.update(run_summary(r["Z"]))
        rows.append(row)
    subj_df = aggregate_subject_summaries(rows)
    out = {}
    for m in PRIMARY_METRICS + SUPPORT_METRICS:
        out[m] = float(pd.to_numeric(subj_df[m], errors="coerce").mean())
    return out


def simulate_mdd_like_runs(ctrl_latent: List[Dict], template_mdd_runs: List[Dict], ar1: Dict[str, np.ndarray], phi_scale: float, noise_scale: float, rng: np.random.Generator) -> List[Dict]:
    out = []
    n_ctrl = len(ctrl_latent)
    for tmpl in template_mdd_runs:
        src = ctrl_latent[rng.integers(0, n_ctrl)]
        start_z = src["Z"][0].copy()
        T = len(tmpl["Z"])
        Z = simulate_ar1(T, start_z, ar1, phi_scale=phi_scale, noise_scale=noise_scale, rng=rng)
        out.append({"pid": tmpl["pid"], "group": "depr", "Z": Z})
    return out


def target_loss(sim_summary: Dict[str, float], target_summary: Dict[str, float], ctrl_sd_map: Dict[str, float]) -> float:
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


def choose_best_candidate(df: pd.DataFrame) -> pd.Series:
    tmp = df.sort_values(
        ["total_score", "primary_loss", "support_loss", "complexity_rank", "perturbation_magnitude", "model_name"],
        ascending=[True, True, True, True, True, True]
    ).reset_index(drop=True)
    return tmp.iloc[0]


def build_candidate_grid() -> pd.DataFrame:
    rows = []
    rows.append({
        "model_name": "baseline",
        "phi_scale": 1.0,
        "noise_scale": 1.0,
        "complexity_rank": 0,
        "perturbation_magnitude": 0.0,
    })
    for p in PHI_SCALE_GRID[1:]:
        rows.append({
            "model_name": "phi_only",
            "phi_scale": float(p),
            "noise_scale": 1.0,
            "complexity_rank": 1,
            "perturbation_magnitude": float((1.0 - p) ** 2),
        })
    for n in NOISE_SCALE_GRID[1:]:
        rows.append({
            "model_name": "noise_only",
            "phi_scale": 1.0,
            "noise_scale": float(n),
            "complexity_rank": 1,
            "perturbation_magnitude": float((n - 1.0) ** 2),
        })
    for p in PHI_SCALE_GRID[1:]:
        for n in NOISE_SCALE_GRID[1:]:
            rows.append({
                "model_name": "phi_plus_noise",
                "phi_scale": float(p),
                "noise_scale": float(n),
                "complexity_rank": 2,
                "perturbation_magnitude": float((1.0 - p) ** 2 + (n - 1.0) ** 2),
            })
    return pd.DataFrame(rows)


def evaluate_candidates(prep: Dict, ar1: Dict[str, np.ndarray]) -> pd.DataFrame:
    cand_df = build_candidate_grid()
    rows = []
    for i, (_, cand) in enumerate(cand_df.iterrows()):
        rng = np.random.default_rng(SEED + 1000 + i)
        sim_runs = simulate_mdd_like_runs(
            ctrl_latent=prep["ctrl_latent"],
            template_mdd_runs=prep["mdd_latent"],
            ar1=ar1,
            phi_scale=float(cand["phi_scale"]),
            noise_scale=float(cand["noise_scale"]),
            rng=rng,
        )
        sim_summary = aggregate_group_summary(sim_runs)
        primary = target_loss(sim_summary, prep["mdd_group"], prep["ctrl_sd_map"])
        support = support_loss(sim_summary, prep["mdd_group"], prep["ctrl_sd_map"])
        pen = LAMBDA_PHI * (1.0 - float(cand["phi_scale"])) ** 2 + LAMBDA_NOISE * (float(cand["noise_scale"]) - 1.0) ** 2
        total = primary + 0.25 * support + pen

        row = cand.to_dict()
        row.update({
            "primary_loss": float(primary),
            "support_loss": float(support),
            "penalty": float(pen),
            "total_score": float(total),
        })
        for m in PRIMARY_METRICS + SUPPORT_METRICS:
            row[f"sim_{m}"] = sim_summary.get(m, np.nan)
            row[f"target_{m}"] = prep["mdd_group"].get(m, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def ppc_for_candidate(candidate: Dict, prep: Dict, ar1: Dict[str, np.ndarray], n_ppc: int = N_PPC) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sim_store = {m: [] for m in PRIMARY_METRICS + SUPPORT_METRICS}
    for rep in range(n_ppc):
        rng = np.random.default_rng(SEED + 500000 + rep)
        sim_runs = simulate_mdd_like_runs(
            ctrl_latent=prep["ctrl_latent"],
            template_mdd_runs=prep["mdd_latent"],
            ar1=ar1,
            phi_scale=float(candidate["phi_scale"]),
            noise_scale=float(candidate["noise_scale"]),
            rng=rng,
        )
        sim_summary = aggregate_group_summary(sim_runs)
        for m in PRIMARY_METRICS + SUPPORT_METRICS:
            sim_store[m].append(sim_summary.get(m, np.nan))

    interval_rows = []
    hits = 0
    metrics = PRIMARY_METRICS + SUPPORT_METRICS
    for m in metrics:
        sv = np.asarray(sim_store[m], dtype=float)
        sv = sv[np.isfinite(sv)]
        av = prep["mdd_group"].get(m, np.nan)
        if len(sv) == 0 or not np.isfinite(av):
            continue
        q025 = float(np.quantile(sv, 0.025))
        q975 = float(np.quantile(sv, 0.975))
        mu = float(np.mean(sv))
        sd = float(np.std(sv, ddof=1)) if len(sv) > 1 else np.nan
        inside = int(q025 <= av <= q975)
        zabs = abs(av - mu) / sd if np.isfinite(sd) and sd > 1e-8 else np.nan
        hits += inside
        interval_rows.append({
            "metric": m,
            "actual_value": av,
            "pred_mean": mu,
            "pred_sd": sd,
            "pred_q025": q025,
            "pred_q975": q975,
            "inside_95": inside,
            "abs_z": zabs,
        })
    interval_df = pd.DataFrame(interval_rows)
    summary_df = pd.DataFrame([{
        "model_name": candidate["model_name"],
        "n_inside": int(hits),
        "n_total": int(len(metrics)),
        "coverage": float(hits / len(metrics)),
        "mean_abs_z": float(pd.to_numeric(interval_df["abs_z"], errors="coerce").mean()) if len(interval_df) else np.nan,
    }])
    return interval_df, summary_df


def control_null_winner_frequency(prep: Dict, ar1: Dict[str, np.ndarray], n_reps: int = N_CONTROL_NULL) -> pd.DataFrame:
    cand_df = build_candidate_grid()
    ctrl_runs = prep["ctrl_latent"]
    winners = []
    for rep in range(n_reps):
        rng = np.random.default_rng(SEED + 900000 + rep)
        null_runs = []
        n_ctrl = len(ctrl_runs)
        for tmpl in ctrl_runs:
            src = ctrl_runs[rng.integers(0, n_ctrl)]
            start_z = src["Z"][0].copy()
            T = len(tmpl["Z"])
            Z = simulate_ar1(T, start_z, ar1, phi_scale=1.0, noise_scale=1.0, rng=rng)
            null_runs.append({"pid": tmpl["pid"], "group": "control", "Z": Z})
        null_summary = aggregate_group_summary(null_runs)

        rows = []
        for _, cand in cand_df.iterrows():
            sim_runs = simulate_mdd_like_runs(
                ctrl_latent=ctrl_runs,
                template_mdd_runs=ctrl_runs,
                ar1=ar1,
                phi_scale=float(cand["phi_scale"]),
                noise_scale=float(cand["noise_scale"]),
                rng=rng,
            )
            sim_summary = aggregate_group_summary(sim_runs)
            primary = target_loss(sim_summary, null_summary, prep["ctrl_sd_map"])
            support = support_loss(sim_summary, null_summary, prep["ctrl_sd_map"])
            pen = LAMBDA_PHI * (1.0 - float(cand["phi_scale"])) ** 2 + LAMBDA_NOISE * (float(cand["noise_scale"]) - 1.0) ** 2
            total = primary + 0.25 * support + pen
            rows.append({
                "model_name": cand["model_name"],
                "phi_scale": float(cand["phi_scale"]),
                "noise_scale": float(cand["noise_scale"]),
                "primary_loss": float(primary),
                "support_loss": float(support),
                "penalty": float(pen),
                "total_score": float(total),
                "complexity_rank": int(cand["complexity_rank"]),
                "perturbation_magnitude": float(cand["perturbation_magnitude"]),
            })
        w = choose_best_candidate(pd.DataFrame(rows))
        winners.append(w["model_name"])

    out = pd.Series(winners).value_counts(normalize=True).rename_axis("model_name").reset_index(name="winner_frequency")
    return out.sort_values("winner_frequency", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_diff_rows = []
    config_overview_rows = []
    consistent_rows = []
    winner_rows = []
    ppc_all_rows = []
    null_rows = []

    for lq_on in LQ_OPTIONS:
        all_runs, best_shape = load_all_runs(apply_lq_correction=lq_on)

        for npc in NPC_OPTIONS:
            config_id = f"npc{npc}_{'LQon' if lq_on else 'LQoff'}"
            cfg_dir = OUT_DIR / config_id
            cfg_dir.mkdir(parents=True, exist_ok=True)

            prep = prepare_latent_runs(all_runs, npc=npc)
            subj_df = prep["subject_df"].copy()
            diff_df = observational_group_diffs(subj_df)

            subj_df.to_csv(cfg_dir / "subject_summaries.csv", index=False)
            diff_df.to_csv(cfg_dir / "group_differences.csv", index=False)
            pd.DataFrame({"selected_pair_name": prep["selected_pairs"]}).to_csv(cfg_dir / "selected_pairs.csv", index=False)

            if len(diff_df):
                tmp = diff_df.copy()
                tmp["config_id"] = config_id
                tmp["npc"] = npc
                tmp["lq_correction"] = bool(lq_on)
                all_diff_rows.append(tmp)

            sig_count = int((pd.to_numeric(diff_df.get("fdr_q", pd.Series(dtype=float)), errors="coerce") < 0.05).sum()) if len(diff_df) else 0
            top_metric = diff_df.iloc[0]["metric"] if len(diff_df) else np.nan
            top_abs_d = float(diff_df.iloc[0]["abs_d"]) if len(diff_df) else np.nan

            config_overview_rows.append({
                "config_id": config_id,
                "npc": npc,
                "lq_correction": bool(lq_on),
                "best_shape": best_shape,
                "n_control_subjects": int((subj_df["group"] == "control").sum()),
                "n_mdd_subjects": int((subj_df["group"] == "depr").sum()),
                "n_selected_pairs": prep["n_selected_pairs"],
                "cum_evr": float(prep["pca_info"]["cum_evr"]),
                "n_metrics_fdr_below_0_05": sig_count,
                "top_metric_by_abs_d": top_metric,
                "top_abs_d": top_abs_d,
            })

            # Stage 2
            ar1 = fit_ar1_model(prep["ctrl_latent"])
            candidate_df = evaluate_candidates(prep, ar1)
            winner = choose_best_candidate(candidate_df)
            interval_df, ppc_df = ppc_for_candidate(dict(winner), prep, ar1, n_ppc=N_PPC)
            null_df = control_null_winner_frequency(prep, ar1, n_reps=N_CONTROL_NULL)

            candidate_df.to_csv(cfg_dir / "ar1_candidate_grid.csv", index=False)
            interval_df.to_csv(cfg_dir / "ar1_ppc_intervals.csv", index=False)
            ppc_df.to_csv(cfg_dir / "ar1_ppc_summary.csv", index=False)
            null_df.to_csv(cfg_dir / "control_null_winner_frequency.csv", index=False)
            with open(cfg_dir / "ar1_model.json", "w") as f:
                json.dump({
                    "phi": [float(v) for v in ar1["phi"]],
                    "sigma": [float(v) for v in ar1["sigma"]],
                    "primary_metrics": PRIMARY_METRICS,
                    "support_metrics": SUPPORT_METRICS,
                }, f, indent=2)

            winner_rows.append({
                "config_id": config_id,
                "npc": npc,
                "lq_correction": bool(lq_on),
                "winner_model_name": winner["model_name"],
                "winner_phi_scale": float(winner["phi_scale"]),
                "winner_noise_scale": float(winner["noise_scale"]),
                "winner_total_score": float(winner["total_score"]),
                "winner_primary_loss": float(winner["primary_loss"]),
                "winner_support_loss": float(winner["support_loss"]),
                "winner_penalty": float(winner["penalty"]),
                "winner_coverage": float(ppc_df["coverage"].iloc[0]),
                "winner_mean_abs_z": float(ppc_df["mean_abs_z"].iloc[0]),
                "actual_ctrl_mean_lag1_autocorr": float(prep["ctrl_group"]["mean_lag1_autocorr"]),
                "actual_mdd_mean_lag1_autocorr": float(prep["mdd_group"]["mean_lag1_autocorr"]),
                "actual_ctrl_lowfreq_fraction": float(prep["ctrl_group"]["lowfreq_fraction"]),
                "actual_mdd_lowfreq_fraction": float(prep["mdd_group"]["lowfreq_fraction"]),
            })

            if len(interval_df):
                tmp = interval_df.copy()
                tmp["config_id"] = config_id
                tmp["winner_model_name"] = winner["model_name"]
                ppc_all_rows.append(tmp)

            if len(null_df):
                tmp = null_df.copy()
                tmp["config_id"] = config_id
                null_rows.append(tmp)

    if not all_diff_rows:
        raise RuntimeError("No observational group-difference results were produced.")

    all_diff_df = pd.concat(all_diff_rows, ignore_index=True)
    overview_df = pd.DataFrame(config_overview_rows)
    winner_df = pd.DataFrame(winner_rows)
    ppc_all_df = pd.concat(ppc_all_rows, ignore_index=True) if len(ppc_all_rows) else pd.DataFrame()
    null_all_df = pd.concat(null_rows, ignore_index=True) if len(null_rows) else pd.DataFrame()

    for metric, sub in all_diff_df.groupby("metric"):
        signs = np.sign(pd.to_numeric(sub["mean_diff_mdd_minus_control"], errors="coerce"))
        pos = int((signs > 0).sum())
        neg = int((signs < 0).sum())
        q = pd.to_numeric(sub["fdr_q"], errors="coerce")
        d = np.abs(pd.to_numeric(sub["cohens_d_mdd_minus_control"], errors="coerce"))
        consistent_rows.append({
            "metric": metric,
            "n_configs": int(len(sub)),
            "n_positive_mdd_minus_control": pos,
            "n_negative_mdd_minus_control": neg,
            "direction_consistent": bool(pos == len(sub) or neg == len(sub)),
            "min_fdr_q": float(np.nanmin(q.values)) if np.isfinite(q).any() else np.nan,
            "median_abs_d": float(np.nanmedian(d.values)) if np.isfinite(d).any() else np.nan,
            "mean_abs_d": float(np.nanmean(d.values)) if np.isfinite(d).any() else np.nan,
        })
    consistent_df = pd.DataFrame(consistent_rows).sort_values(
        ["direction_consistent", "min_fdr_q", "median_abs_d"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    winner_consensus = winner_df["winner_model_name"].value_counts().rename_axis("winner_model_name").reset_index(name="n_wins")

    overview_df.to_csv(OUT_DIR / "config_overview.csv", index=False)
    all_diff_df.to_csv(OUT_DIR / "all_group_differences.csv", index=False)
    consistent_df.to_csv(OUT_DIR / "consistent_metrics_summary.csv", index=False)
    winner_df.to_csv(OUT_DIR / "ar1_winner_by_config.csv", index=False)
    winner_consensus.to_csv(OUT_DIR / "ar1_winner_consensus.csv", index=False)
    if len(ppc_all_df):
        ppc_all_df.to_csv(OUT_DIR / "ar1_ppc_intervals_all_configs.csv", index=False)
    if len(null_all_df):
        null_all_df.to_csv(OUT_DIR / "control_null_winner_frequency_all_configs.csv", index=False)

    lines = []
    lines.append("mdd_observation_ar1 summary")
    lines.append("")
    lines.append("Primary question:")
    lines.append("What basic dynamical features most reproducibly differ in MDD?")
    lines.append("")
    lines.append("Stage 1:")
    lines.append("Simple control vs MDD observational comparison")
    lines.append("")
    lines.append("Stage 2:")
    lines.append("AR(1) sufficiency test for mean_lag1_autocorr and lowfreq_fraction")
    lines.append("")
    lines.append("Config overview:")
    lines.append(overview_df.to_string(index=False))
    lines.append("")
    lines.append("Consistency summary:")
    lines.append(consistent_df.to_string(index=False))
    lines.append("")
    lines.append("AR(1) winner consensus:")
    lines.append(winner_consensus.to_string(index=False))
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()

