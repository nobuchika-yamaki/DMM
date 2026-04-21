# mdd_observation_ar1_ho_split_v2.py
# Supplementary robustness analysis using the Harvard-Oxford cortical atlas
# with left-right symmetric splitting.
#
# This version fixes a bug in the previous script:
# atlas ROI-pair counts can differ slightly across runs after resampling and voxel-thresholding.
# The present version harmonizes runs to the intersection of ROI-pair names before PCA and AR(1).

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from nilearn import datasets

import matplotlib
matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import mdd_minimal_mapping as base
except ImportError as exc:
    raise ImportError("mdd_minimal_mapping.py must be in the same folder as this script.") from exc

find_niftis = base.find_niftis
get_pid = base.get_pid
find_tsv = base.find_tsv
fit_pca = base.fit_pca
project_pca = base.project_pca
run_summary = base.run_summary
aggregate_subject_summaries = base.aggregate_subject_summaries
norm_group = base.norm_group
INPUT_ROOT = base.INPUT_ROOT
SUMMARY_METRICS = base.SUMMARY_METRICS

HOME = Path.home()
OUT_DIR = HOME / "Desktop" / "mdd_observation_ar1_ho_split_results_v2"

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
MIN_VOXELS_PER_SIDE = 30
MAX_TOP_PAIRS = 24


def zscore1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


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


def stable_sd(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 1.0
    s = float(np.std(x, ddof=1))
    return s if np.isfinite(s) and s > 1e-6 else 1.0


def _fetch_ho_symmetric():
    try:
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", symmetric_split=True)
        maps = atlas.maps
        if isinstance(maps, nib.Nifti1Image):
            img = maps
        else:
            img = nib.load(str(maps))
        labels = list(atlas.labels)
        return img, labels, "nilearn_symmetric_split"
    except TypeError:
        return None, None, None
    except Exception:
        return None, None, None


def _manual_split_ho() -> Tuple[nib.Nifti1Image, List[str], str]:
    atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    maps = atlas.maps
    if isinstance(maps, nib.Nifti1Image):
        atlas_img = maps
    else:
        atlas_img = nib.load(str(maps))
    labels = list(atlas.labels)

    arr = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    nx, ny, nz = arr.shape
    ijk = np.indices((nx, ny, nz)).reshape(3, -1).T
    xyz = nib.affines.apply_affine(atlas_img.affine, ijk)
    x_world = xyz[:, 0].reshape(nx, ny, nz)

    out = np.zeros_like(arr, dtype=np.int32)
    new_labels = ["Background"]
    next_id = 1

    for old_id in sorted(int(v) for v in np.unique(arr) if int(v) != 0):
        mask = arr == old_id
        if not np.any(mask):
            continue
        base_name = labels[old_id] if old_id < len(labels) else f"Label_{old_id}"

        left_mask = mask & (x_world < 0)
        right_mask = mask & (x_world >= 0)

        if np.any(left_mask):
            out[left_mask] = next_id
            new_labels.append(f"{base_name}_L")
            next_id += 1
        if np.any(right_mask):
            out[right_mask] = next_id
            new_labels.append(f"{base_name}_R")
            next_id += 1

    return nib.Nifti1Image(out, atlas_img.affine, atlas_img.header), new_labels, "manual_midline_split"


def fetch_harvard_oxford_symmetric():
    img, labels, mode = _fetch_ho_symmetric()
    if img is not None:
        return img, labels, mode
    return _manual_split_ho()


def resample_label_atlas_to_bold(atlas_img: nib.Nifti1Image, bold_img: nib.Nifti1Image) -> nib.Nifti1Image:
    target = (bold_img.shape[:3], bold_img.affine)
    res = resample_from_to(atlas_img, target, order=0)
    arr = np.rint(res.get_fdata()).astype(np.int32)
    return nib.Nifti1Image(arr, res.affine, res.header)


def labels_to_name_map(labels: List[str]) -> Dict[int, str]:
    return {i: str(name) for i, name in enumerate(labels)}


def normalize_lr_name(name: str) -> Tuple[str, str | None]:
    n = str(name).strip()
    if n.endswith("_L"):
        return n[:-2], "L"
    if n.endswith("_R"):
        return n[:-2], "R"
    low = n.lower()
    if low.startswith("left "):
        return n[5:].strip(), "L"
    if low.startswith("right "):
        return n[6:].strip(), "R"
    if low.endswith(" left"):
        return n[:-5].strip(), "L"
    if low.endswith(" right"):
        return n[:-6].strip(), "R"
    return n, None


def build_ho_pairs(label_names: Dict[int, str]) -> List[Dict]:
    buckets: Dict[str, Dict[str, Dict]] = {}
    for idx, name in label_names.items():
        if idx == 0:
            continue
        stem, side = normalize_lr_name(name)
        if side is None:
            continue
        buckets.setdefault(stem, {})
        buckets[stem][side] = {"id": int(idx), "name": str(name)}

    pairs = []
    for stem, sides in sorted(buckets.items()):
        if "L" in sides and "R" in sides:
            pairs.append({
                "pair_name": stem,
                "left_id": int(sides["L"]["id"]),
                "left_name": str(sides["L"]["name"]),
                "right_id": int(sides["R"]["id"]),
                "right_name": str(sides["R"]["name"]),
            })
    return pairs


def load_run_ho(nifti_path: Path, atlas_img: nib.Nifti1Image, ho_pairs: List[Dict], lq_sign: float = 1.0):
    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D, got {data.ndim}D: {nifti_path}")

    nx, ny, nz, nt = data.shape
    flat = data.reshape(-1, nt)
    valid_voxel = np.all(np.isfinite(flat), axis=1) & np.any(flat != 0, axis=1)

    atlas_res = resample_label_atlas_to_bold(atlas_img, img)
    atlas_arr = np.asarray(atlas_res.get_fdata(), dtype=np.int32)
    atlas_flat = atlas_arr.reshape(-1)

    pair_names = []
    diff_ts_list = []
    pair_info_rows = []

    for p in ho_pairs:
        left_idx = np.where(atlas_flat == p["left_id"])[0]
        right_idx = np.where(atlas_flat == p["right_id"])[0]

        left_idx = left_idx[valid_voxel[left_idx]]
        right_idx = right_idx[valid_voxel[right_idx]]

        if len(left_idx) < MIN_VOXELS_PER_SIDE or len(right_idx) < MIN_VOXELS_PER_SIDE:
            continue

        left_ts = flat[left_idx].mean(axis=0)
        right_ts = flat[right_idx].mean(axis=0)

        diff_ts = lq_sign * (zscore1d(right_ts) - zscore1d(left_ts))
        pair_names.append(p["pair_name"])
        diff_ts_list.append(diff_ts)
        pair_info_rows.append({
            "pair_name": p["pair_name"],
            "left_name": p["left_name"],
            "right_name": p["right_name"],
            "left_id": p["left_id"],
            "right_id": p["right_id"],
            "left_voxels": int(len(left_idx)),
            "right_voxels": int(len(right_idx)),
        })

    if len(diff_ts_list) < 2:
        raise ValueError(f"Too few valid atlas ROI pairs: {nifti_path.name}")

    D = np.column_stack(diff_ts_list)
    roi_df = pd.DataFrame(pair_info_rows)
    return D, pair_names, f"{nx}x{ny}x{nz}", roi_df


def harmonize_runs_to_common_pairs(runs: List[Dict]) -> Tuple[List[Dict], List[str]]:
    common = None
    for r in runs:
        names = set(r["names"])
        common = names if common is None else common.intersection(names)
    common_names = sorted(common) if common else []
    if len(common_names) < 2:
        raise RuntimeError("Too few common ROI pairs across runs after atlas resampling.")
    out = []
    for r in runs:
        name_to_idx = {name: i for i, name in enumerate(r["names"])}
        idx = [name_to_idx[n] for n in common_names]
        rr = dict(r)
        rr["D"] = r["D"][:, idx]
        rr["names"] = common_names
        out.append(rr)
    return out, common_names


def select_top_pairs_by_control_variance(ctrl_runs: List[Dict], pair_names: List[str], top_n: int = MAX_TOP_PAIRS):
    P = len(pair_names)
    pair_vars = np.zeros(P)
    for r in ctrl_runs:
        pair_vars += np.var(r["D"], axis=0, ddof=1)
    pair_vars /= max(1, len(ctrl_runs))
    top_idx = np.argsort(pair_vars)[::-1][:min(top_n, P)]
    top_idx = sorted(top_idx.tolist())
    return top_idx, [pair_names[i] for i in top_idx]


def load_all_runs_ho(apply_lq_correction: bool, atlas_img: nib.Nifti1Image, ho_pairs: List[Dict]):
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
    roi_info_rows = []

    for p in niftis:
        pid = get_pid(p)
        group = pid_to_group.get(pid, "unknown")
        lq_sign = lq_map.get(pid, 1.0)
        try:
            D, names, shape, roi_df = load_run_ho(p, atlas_img=atlas_img, ho_pairs=ho_pairs, lq_sign=lq_sign)
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
        if len(roi_df):
            tmp = roi_df.copy()
            tmp["file"] = str(p)
            tmp["pid"] = pid
            roi_info_rows.append(tmp)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    if not runs:
        raise RuntimeError("No analysable atlas-based runs were loaded.")

    best_shape = max(shape_counts, key=shape_counts.get)
    runs = [r for r in runs if r["shape"] == best_shape]

    runs, common_names = harmonize_runs_to_common_pairs(runs)

    roi_info_df = pd.concat(roi_info_rows, ignore_index=True) if roi_info_rows else pd.DataFrame()
    return runs, best_shape, roi_info_df, common_names


def prepare_latent_runs(all_runs: List[Dict], npc: int):
    pair_names = all_runs[0]["names"]
    ctrl_runs = [r for r in all_runs if r["group"] == "control"]
    depr_runs = [r for r in all_runs if r["group"] == "depr"]

    if len(ctrl_runs) < 5 or len(depr_runs) < 5:
        raise RuntimeError("Too few control or depression runs.")

    sel_idx, sel_names = select_top_pairs_by_control_variance(ctrl_runs, pair_names, top_n=MAX_TOP_PAIRS)
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
    depr_subj = subj_df[subj_df["group"] == "depr"].copy()

    ctrl_group = {}
    depr_group = {}
    ctrl_sd_map = {}
    for m in PRIMARY_METRICS + SUPPORT_METRICS:
        ctrl_group[m] = float(pd.to_numeric(ctrl_subj[m], errors="coerce").mean())
        depr_group[m] = float(pd.to_numeric(depr_subj[m], errors="coerce").mean())
        ctrl_sd_map[m] = stable_sd(pd.to_numeric(ctrl_subj[m], errors="coerce").values)

    return {
        "ctrl_latent": ctrl_latent,
        "depr_latent": depr_latent,
        "subject_df": subj_df,
        "ctrl_group": ctrl_group,
        "depr_group": depr_group,
        "ctrl_sd_map": ctrl_sd_map,
        "pca_info": pca_info,
        "n_selected_pairs": int(len(sel_idx)),
        "selected_pairs": sel_names,
        "n_common_pairs": int(len(pair_names)),
    }


def observational_group_diffs(subj_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(SUMMARY_METRICS):
        x = pd.to_numeric(subj_df.loc[subj_df["group"] == "control", m], errors="coerce").dropna().values
        y = pd.to_numeric(subj_df.loc[subj_df["group"] == "depr", m], errors="coerce").dropna().values
        if len(x) < 2 or len(y) < 2:
            continue
        md, lo, hi = bootstrap_mean_diff(x, y, n_boot=N_BOOT, seed=SEED + i)
        pp = permutation_pvalue(x, y, n_perm=N_PERM, seed=SEED + 10000 + i)
        rows.append({
            "metric": m,
            "n_control": len(x),
            "n_depression": len(y),
            "control_mean": float(np.mean(x)),
            "control_sd": float(np.std(x, ddof=1)),
            "depression_mean": float(np.mean(y)),
            "depression_sd": float(np.std(y, ddof=1)),
            "mean_diff_depression_minus_control": float(np.mean(y) - np.mean(x)),
            "bootstrap_mean_diff": md,
            "ci_2_5": lo,
            "ci_97_5": hi,
            "cohens_d_depression_minus_control": cohens_d(y, x),
            "perm_p": pp,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr_q"] = bh_fdr(out["perm_p"].values)
        out["abs_d"] = np.abs(out["cohens_d_depression_minus_control"])
        out = out.sort_values(["fdr_q", "abs_d"], ascending=[True, False]).reset_index(drop=True)
    return out


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


def simulate_one_run_ar1(length: int, z0: np.ndarray, ar1: Dict[str, np.ndarray], phi_scale: float, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    phi = np.asarray(ar1["phi"], dtype=float) * float(phi_scale)
    sigma = np.asarray(ar1["sigma"], dtype=float) * float(noise_scale)
    D = len(phi)
    Z = np.zeros((length, D), dtype=float)
    Z[0] = np.asarray(z0, dtype=float)
    for t in range(1, length):
        eps = rng.normal(0.0, sigma, size=D)
        Z[t] = phi * Z[t - 1] + eps
    return np.clip(Z, -10.0, 10.0)


def simulate_depression_like_runs(ctrl_latent: List[Dict], template_depr_runs: List[Dict], ar1: Dict[str, np.ndarray], phi_scale: float, noise_scale: float, rng: np.random.Generator) -> List[Dict]:
    out = []
    n_ctrl = len(ctrl_latent)
    for tmpl in template_depr_runs:
        src = ctrl_latent[rng.integers(0, n_ctrl)]
        z0 = src["Z"][0].copy()
        T = len(tmpl["Z"])
        Z = simulate_one_run_ar1(T, z0, ar1, phi_scale=phi_scale, noise_scale=noise_scale, rng=rng)
        out.append({"pid": tmpl["pid"], "group": "depr", "Z": Z})
    return out


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
    rows.append({"model_name": "baseline", "phi_scale": 1.0, "noise_scale": 1.0, "complexity_rank": 0, "perturbation_magnitude": 0.0})
    for p in PHI_SCALE_GRID[1:]:
        rows.append({"model_name": "phi_only", "phi_scale": float(p), "noise_scale": 1.0, "complexity_rank": 1, "perturbation_magnitude": float((1.0 - p) ** 2)})
    for n in NOISE_SCALE_GRID[1:]:
        rows.append({"model_name": "noise_only", "phi_scale": 1.0, "noise_scale": float(n), "complexity_rank": 1, "perturbation_magnitude": float((n - 1.0) ** 2)})
    for p in PHI_SCALE_GRID[1:]:
        for n in NOISE_SCALE_GRID[1:]:
            rows.append({"model_name": "phi_plus_noise", "phi_scale": float(p), "noise_scale": float(n), "complexity_rank": 2, "perturbation_magnitude": float((1.0 - p) ** 2 + (n - 1.0) ** 2)})
    return pd.DataFrame(rows)


def evaluate_candidates(prep: Dict, ar1: Dict[str, np.ndarray]) -> pd.DataFrame:
    cand_df = build_candidate_grid()
    rows = []
    for i, (_, cand) in enumerate(cand_df.iterrows()):
        rng = np.random.default_rng(SEED + 1000 + i)
        sim_runs = simulate_depression_like_runs(prep["ctrl_latent"], prep["depr_latent"], ar1, float(cand["phi_scale"]), float(cand["noise_scale"]), rng)
        sim_summary = aggregate_group_summary(sim_runs)
        primary = target_loss(sim_summary, prep["depr_group"], prep["ctrl_sd_map"])
        support = support_loss(sim_summary, prep["depr_group"], prep["ctrl_sd_map"])
        pen = LAMBDA_PHI * (1.0 - float(cand["phi_scale"])) ** 2 + LAMBDA_NOISE * (float(cand["noise_scale"]) - 1.0) ** 2
        total = primary + 0.25 * support + pen
        row = cand.to_dict()
        row.update({"primary_loss": float(primary), "support_loss": float(support), "penalty": float(pen), "total_score": float(total)})
        for m in PRIMARY_METRICS + SUPPORT_METRICS:
            row[f"sim_{m}"] = sim_summary.get(m, np.nan)
            row[f"target_{m}"] = prep["depr_group"].get(m, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def ppc_for_candidate(prep: Dict, ar1: Dict[str, np.ndarray], best_row: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    for rep in range(N_PPC):
        rng = np.random.default_rng(SEED + 20000 + rep)
        sim_runs = simulate_depression_like_runs(prep["ctrl_latent"], prep["depr_latent"], ar1, float(best_row["phi_scale"]), float(best_row["noise_scale"]), rng)
        sim_summary = aggregate_group_summary(sim_runs)
        sim_summary["rep"] = rep
        summaries.append(sim_summary)

    sim_df = pd.DataFrame(summaries)
    for m in PRIMARY_METRICS + SUPPORT_METRICS:
        vals = pd.to_numeric(sim_df[m], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        actual = prep["depr_group"].get(m, np.nan)
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        lo = float(np.quantile(vals, 0.025))
        hi = float(np.quantile(vals, 0.975))
        z = float(abs(actual - mu) / sd) if np.isfinite(actual) and np.isfinite(sd) and sd > 1e-8 else np.nan
        rows.append({"metric": m, "actual": actual, "sim_mean": mu, "sim_sd": sd, "ppc_lo_2_5": lo, "ppc_hi_97_5": hi, "covered": bool(np.isfinite(actual) and lo <= actual <= hi), "abs_z": z})
    ppc_df = pd.DataFrame(rows)
    return ppc_df, sim_df


def control_null_winner_frequency(prep: Dict, ar1: Dict[str, np.ndarray]) -> pd.DataFrame:
    cand_df = build_candidate_grid()
    counts = {name: 0 for name in cand_df["model_name"].unique()}
    ctrl_runs = prep["ctrl_latent"]
    n_ctrl = len(ctrl_runs)

    for rep in range(N_CONTROL_NULL):
        rng = np.random.default_rng(SEED + 40000 + rep)
        null_runs = []
        for tmpl in ctrl_runs:
            src = ctrl_runs[rng.integers(0, n_ctrl)]
            z0 = src["Z"][0].copy()
            T = len(tmpl["Z"])
            Z = simulate_one_run_ar1(T, z0, ar1, phi_scale=1.0, noise_scale=1.0, rng=rng)
            null_runs.append({"pid": tmpl["pid"], "group": "control", "Z": Z})
        target_summary = aggregate_group_summary(null_runs)

        rows = []
        for i, (_, cand) in enumerate(cand_df.iterrows()):
            sim_runs = []
            rng2 = np.random.default_rng(SEED + 50000 + rep * 1000 + i)
            for tmpl in ctrl_runs:
                src = ctrl_runs[rng2.integers(0, n_ctrl)]
                z0 = src["Z"][0].copy()
                T = len(tmpl["Z"])
                Z = simulate_one_run_ar1(T, z0, ar1, float(cand["phi_scale"]), float(cand["noise_scale"]), rng2)
                sim_runs.append({"pid": tmpl["pid"], "group": "control", "Z": Z})
            sim_summary = aggregate_group_summary(sim_runs)
            primary = target_loss(sim_summary, target_summary, prep["ctrl_sd_map"])
            support = support_loss(sim_summary, target_summary, prep["ctrl_sd_map"])
            pen = LAMBDA_PHI * (1.0 - float(cand["phi_scale"])) ** 2 + LAMBDA_NOISE * (float(cand["noise_scale"]) - 1.0) ** 2
            total = primary + 0.25 * support + pen

            row = cand.to_dict()
            row.update({"primary_loss": float(primary), "support_loss": float(support), "penalty": float(pen), "total_score": float(total)})
            rows.append(row)
        rep_df = pd.DataFrame(rows)
        win = choose_best_candidate(rep_df)
        counts[str(win["model_name"])] += 1

    out = pd.DataFrame({"model_name": list(counts.keys()), "winner_frequency": [counts[k] / N_CONTROL_NULL for k in counts.keys()]})
    return out.sort_values("winner_frequency", ascending=False).reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    atlas_img_raw, atlas_labels, split_mode = fetch_harvard_oxford_symmetric()
    label_map = labels_to_name_map(atlas_labels)
    ho_pairs = build_ho_pairs(label_map)
    if len(ho_pairs) < 5:
        raise RuntimeError("Too few left-right homologous Harvard-Oxford pairs were detected.")

    overview_rows = []
    all_diff_rows = []
    winner_rows = []
    ppc_all = []
    null_all = []
    roi_info_all = []

    for lq_on in LQ_OPTIONS:
        all_runs, best_shape, roi_info_df, common_names = load_all_runs_ho(
            apply_lq_correction=lq_on,
            atlas_img=atlas_img_raw,
            ho_pairs=ho_pairs,
        )
        if len(roi_info_df):
            roi_info_df["lq_correction"] = bool(lq_on)
            roi_info_all.append(roi_info_df)

        for npc in NPC_OPTIONS:
            config_id = f"npc{npc}_{'LQon' if lq_on else 'LQoff'}"
            prep = prepare_latent_runs(all_runs, npc=npc)
            ar1 = fit_ar1_model(prep["ctrl_latent"])

            obs_diff = observational_group_diffs(prep["subject_df"])
            if len(obs_diff):
                obs_diff["npc"] = npc
                obs_diff["lq_correction"] = bool(lq_on)
                obs_diff["config_id"] = config_id
                all_diff_rows.append(obs_diff)

            cand_df = evaluate_candidates(prep, ar1)
            best = choose_best_candidate(cand_df)

            ppc_df, _ = ppc_for_candidate(prep, ar1, best)
            if len(ppc_df):
                ppc_df["npc"] = npc
                ppc_df["lq_correction"] = bool(lq_on)
                ppc_df["config_id"] = config_id
                ppc_all.append(ppc_df)

            null_df = control_null_winner_frequency(prep, ar1)
            if len(null_df):
                null_df["npc"] = npc
                null_df["lq_correction"] = bool(lq_on)
                null_df["config_id"] = config_id
                null_all.append(null_df)

            winner_coverage = float(np.mean(ppc_df["covered"].astype(float))) if len(ppc_df) else np.nan
            winner_abs_z = float(np.nanmean(pd.to_numeric(ppc_df["abs_z"], errors="coerce").values)) if len(ppc_df) else np.nan

            overview_rows.append({
                "config_id": config_id,
                "npc": npc,
                "lq_correction": bool(lq_on),
                "best_shape": best_shape,
                "n_control_subjects": int((prep["subject_df"]["group"] == "control").sum()),
                "n_depression_subjects": int((prep["subject_df"]["group"] == "depr").sum()),
                "n_common_pairs": int(prep["n_common_pairs"]),
                "n_selected_pairs": int(prep["n_selected_pairs"]),
                "cum_evr": float(prep["pca_info"]["cum_evr"]),
                "atlas_pair_count_available": int(len(ho_pairs)),
                "atlas_type": "HarvardOxford_cortical",
                "split_mode": split_mode,
            })

            winner_rows.append({
                "config_id": config_id,
                "npc": npc,
                "lq_correction": bool(lq_on),
                "winner_model_name": str(best["model_name"]),
                "winner_phi_scale": float(best["phi_scale"]),
                "winner_noise_scale": float(best["noise_scale"]),
                "winner_primary_loss": float(best["primary_loss"]),
                "winner_support_loss": float(best["support_loss"]),
                "winner_total_score": float(best["total_score"]),
                "winner_coverage": winner_coverage,
                "winner_mean_abs_z": winner_abs_z,
                "actual_ctrl_mean_lag1_autocorr": float(prep["ctrl_group"]["mean_lag1_autocorr"]),
                "actual_depr_mean_lag1_autocorr": float(prep["depr_group"]["mean_lag1_autocorr"]),
                "actual_ctrl_lowfreq_fraction": float(prep["ctrl_group"]["lowfreq_fraction"]),
                "actual_depr_lowfreq_fraction": float(prep["depr_group"]["lowfreq_fraction"]),
                "sim_mean_lag1_autocorr": float(best.get("sim_mean_lag1_autocorr", np.nan)),
                "sim_lowfreq_fraction": float(best.get("sim_lowfreq_fraction", np.nan)),
            })

    overview_df = pd.DataFrame(overview_rows).sort_values(["npc", "lq_correction"]).reset_index(drop=True)
    all_diff_df = pd.concat(all_diff_rows, ignore_index=True) if all_diff_rows else pd.DataFrame()
    winner_df = pd.DataFrame(winner_rows).sort_values(["npc", "lq_correction"]).reset_index(drop=True)
    ppc_all_df = pd.concat(ppc_all, ignore_index=True) if ppc_all else pd.DataFrame()
    null_all_df = pd.concat(null_all, ignore_index=True) if null_all else pd.DataFrame()
    roi_all_df = pd.concat(roi_info_all, ignore_index=True) if roi_info_all else pd.DataFrame()

    consistent_rows = []
    if len(all_diff_df):
        for metric, sub in all_diff_df.groupby("metric"):
            signs = np.sign(pd.to_numeric(sub["mean_diff_depression_minus_control"], errors="coerce"))
            pos = int((signs > 0).sum())
            neg = int((signs < 0).sum())
            q = pd.to_numeric(sub["fdr_q"], errors="coerce")
            d = np.abs(pd.to_numeric(sub["cohens_d_depression_minus_control"], errors="coerce"))
            consistent_rows.append({
                "metric": metric,
                "n_configs": int(len(sub)),
                "n_positive_depression_minus_control": pos,
                "n_negative_depression_minus_control": neg,
                "direction_consistent": bool(pos == len(sub) or neg == len(sub)),
                "min_fdr_q": float(np.nanmin(q.values)) if np.isfinite(q).any() else np.nan,
                "median_abs_d": float(np.nanmedian(d.values)) if np.isfinite(d).any() else np.nan,
                "mean_abs_d": float(np.nanmean(d.values)) if np.isfinite(d).any() else np.nan,
            })
    consistent_df = pd.DataFrame(consistent_rows).sort_values(["direction_consistent", "min_fdr_q", "median_abs_d"], ascending=[False, True, False]).reset_index(drop=True) if consistent_rows else pd.DataFrame()
    winner_consensus = winner_df["winner_model_name"].value_counts().rename_axis("winner_model_name").reset_index(name="n_wins") if len(winner_df) else pd.DataFrame()

    overview_df.to_csv(OUT_DIR / "config_overview.csv", index=False)
    all_diff_df.to_csv(OUT_DIR / "all_group_differences.csv", index=False)
    consistent_df.to_csv(OUT_DIR / "consistent_metrics_summary.csv", index=False)
    winner_df.to_csv(OUT_DIR / "ar1_winner_by_config.csv", index=False)
    winner_consensus.to_csv(OUT_DIR / "ar1_winner_consensus.csv", index=False)
    if len(ppc_all_df):
        ppc_all_df.to_csv(OUT_DIR / "ar1_ppc_intervals_all_configs.csv", index=False)
    if len(null_all_df):
        null_all_df.to_csv(OUT_DIR / "control_null_winner_frequency_all_configs.csv", index=False)
    if len(roi_all_df):
        roi_all_df.to_csv(OUT_DIR / "ho_roi_pairs_manifest.csv", index=False)

    lines = []
    lines.append("mdd_observation_ar1_ho_split_v2 summary")
    lines.append("")
    lines.append("Purpose:")
    lines.append("Supplementary atlas-based sensitivity analysis using Harvard-Oxford cortical atlas with left-right splitting.")
    lines.append("This tests whether the main directional findings are preserved under an anatomical atlas.")
    lines.append("")
    lines.append(f"Atlas split mode: {split_mode}")
    lines.append("")
    lines.append("Config overview:")
    lines.append(overview_df.to_string(index=False))
    lines.append("")
    if len(consistent_df):
        lines.append("Consistency summary:")
        lines.append(consistent_df.to_string(index=False))
        lines.append("")
    if len(winner_consensus):
        lines.append("AR(1) winner consensus:")
        lines.append(winner_consensus.to_string(index=False))
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
