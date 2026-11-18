#!/usr/bin/env python3
"""Main analysis pipeline for the gaze-aware localized windshield study.

Runs the six randomized scenario families with matched controllers, the
accuracy-smoothness tuning sweep, the dual-eye containment Monte Carlo, the
statistical comparison suite, and the latency/filter checks. All outputs
(CSV/JSON) are written under outputs/.

Everything is synthetic simulation; see src/core.py for the geometry.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT.parent / "outputs"
OUT.mkdir(exist_ok=True)

import core  # noqa: E402

CONTROLLERS = ["fixed", "crisp", "fuzzy"]
STAGES = ["raw", "gated", "filtered", "full"]


def seed_list():
    return {fam: [core.FAMILY_BASE_SEED + 1000 * i + k
                  for k in range(core.SEEDS_PER_FAMILY)]
            for i, fam in enumerate(core.FAMILIES)}


def run_main_study(seeds):
    rows, traces = [], []
    for fam in core.FAMILIES:
        for seed in seeds[fam]:
            sc = core.generate_scenario(fam, seed)
            for ctrl in CONTROLLERS:
                raw = core.controller_raw(ctrl, sc["g_meas"], sc["a_meas"],
                                          sc["c_meas"])
                alpha_full, filt, state = core.temporal_pipeline(
                    raw, sc["c_meas"], sc["inhibit"].astype(bool))
                alpha_gated = np.where(sc["c_meas"] >= core.GAMMA, raw, 0.0)
                alpha_filt, _, _ = core.temporal_pipeline(
                    raw, sc["c_meas"], sc["inhibit"].astype(bool),
                    hysteresis=False)
                for stage, alpha in (("raw", raw), ("gated", alpha_gated),
                                     ("filtered", alpha_filt), ("full", alpha_full)):
                    rows.append(core.compute_metrics(sc, alpha, ctrl, stage))
            if seed == seeds[fam][0]:
                traces.append(sc)
    return pd.DataFrame(rows), traces


def bootstrap_ci_diff(x, y, n_boot=2000, seed=7):
    """95% bootstrap CI of the median paired difference x - y."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(len(x), len(y))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = np.median(x[idx] - y[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def bootstrap_ci_median(v, n_boot=2000, seed=11):
    """95% bootstrap CI of the median of a single sample."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    meds = np.empty(n_boot)
    for b in range(n_boot):
        meds[b] = np.median(v[rng.integers(0, len(v), len(v))])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def paired_stats(df, fam, metric, a="fuzzy", b="crisp"):
    """Wilcoxon signed-rank + effect size between two full-stage controllers."""
    from scipy import stats as st
    dsub = df if fam == "ALL" else df[df.family == fam]
    da = dsub[(dsub.controller == a) & (dsub.stage == "full")].set_index("seed")[metric]
    db = dsub[(dsub.controller == b) & (dsub.stage == "full")].set_index("seed")[metric]
    common = da.index.intersection(db.index)
    x, y = da.loc[common].to_numpy(), db.loc[common].to_numpy()
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 10 or np.all(x == y):
        return {"family": fam, "metric": metric, "n": len(x), "p_value": np.nan,
                "median_diff": np.nan, "rank_biserial": np.nan,
                "pct_seeds_a_better": np.nan}
    res = st.wilcoxon(x, y)
    rb = 1 - 2 * res.statistic / (len(x) * (len(x) + 1) / 2)   # rank-biserial (x better)
    return {"family": fam, "metric": metric, "n": len(x),
            "p_value": float(res.pvalue),
            "median_diff": float(np.median(x - y)),
            "rank_biserial": float(rb),
            "pct_seeds_a_better": float(np.mean(x < y) * 100)}


def run_statistics(df):
    rows = []
    for fam in core.FAMILIES + ["ALL"]:
        for ctrl in CONTROLLERS:
            for metric in ["warranted_attenuation", "nuisance_attenuation",
                           "switch_count", "total_variation",
                           "activation_delay_s", "release_delay_s"]:
                d = df if fam == "ALL" else df[df.family == fam]
                v = d[(d.controller == ctrl) & (d.stage == "full")][metric].dropna()
                lo, hi = bootstrap_ci_median(v)  # CI of the median itself
                rows.append({"family": fam, "controller": ctrl, "metric": metric,
                             "n": len(v), "median": float(v.median()),
                             "iqr": float(v.quantile(0.75) - v.quantile(0.25)),
                             "ci_lo": lo, "ci_hi": hi})
    summ = pd.DataFrame(rows)
    comp = pd.DataFrame([paired_stats(df, fam, m)
                         for fam in core.FAMILIES + ["ALL"]
                         for m in ["warranted_attenuation", "nuisance_attenuation",
                                   "total_variation", "switch_count"]])
    return summ, comp


def run_tradeoff(seeds, taus=None):
    """Matched tau sweep for crisp and fuzzy with proportional hysteresis."""
    if taus is None:
        taus = np.geomspace(0.05, 1.0, 10)
    rows = []
    for fam in core.FAMILIES:
        use_seeds = seeds[fam][:60]        # subset keeps the sweep tractable
        for seed in use_seeds:
            sc = core.generate_scenario(fam, seed)
            for ctrl in ("crisp", "fuzzy"):
                raw = core.controller_raw(ctrl, sc["g_meas"], sc["a_meas"],
                                          sc["c_meas"])
                for tau in taus:
                    th_on = min(0.5, max(0.02, 0.5 * tau))
                    th_off = th_on / 3.0
                    alpha, _, _ = core.temporal_pipeline(
                        raw, sc["c_meas"], sc["inhibit"].astype(bool),
                        tau=float(tau), theta_on=th_on, theta_off=th_off)
                    m = core.compute_metrics(sc, alpha, ctrl, f"tau={tau:.3f}")
                    m["tau_s"] = float(tau)
                    rows.append(m)
    return pd.DataFrame(rows)


def containment_monte_carlo(n_mc=200_000, seed=99):
    """Dual-eye containment using predicted zone centres and true intercepts.

    For each Monte Carlo draw, the two predicted zone centres are the
    calibrated eye-to-source intercepts displaced by tracking error (and any
    common bias). The two true intercepts remain at the calibrated locations
    for static-error cases. For latency cases the true intercepts move while
    predicted centres retain independent static tracking error. Success means
    BOTH true eye intercepts lie in the UNION of the two predicted circles.

    This is the physical geometry in Sections 3.3 and 6.4. In particular,
    error vectors are NOT themselves points tested against circles fixed at
    +/- S_IO/2; that coordinate mix-up was the source of the previous
    containment discrepancy.
    """
    rng = np.random.default_rng(seed)
    half = core.S_IO / 2
    nominal = np.array([[-half, 0.0], [half, 0.0]])  # L, R intercepts
    sig = 0.015
    radii = [0.0514, 0.060]

    def success(true_pts, predicted_centres, r):
        # arrays: (n, 2 eyes, 2 coordinates); each true point may be
        # contained by either predicted circle, as specified by the union.
        delta = true_pts[:, :, None, :] - predicted_centres[:, None, :, :]
        d2 = np.sum(delta * delta, axis=-1)
        return np.all(np.min(d2, axis=2) <= r * r, axis=1)

    def margin_and_area(true_pts, pred, r0, target=0.95):
        delta = true_pts[:, :, None, :] - pred[:, None, :, :]
        d2 = np.sum(delta * delta, axis=-1)
        worst = np.max(np.min(d2, axis=2), axis=1)  # both eyes contained
        grid = np.linspace(0.0, 0.12, 241)
        probs = np.array([np.mean(worst <= (r0 + m) ** 2) for m in grid])
        margin = float(grid[np.argmax(probs >= target)]) if np.any(probs >= target) else float('nan')
        area_cost = (core.union_area(r0 + margin) - core.union_area(r0)
                     if np.isfinite(margin) else float('nan'))
        return float(np.mean(worst <= r0 ** 2)), margin, float(area_cost)

    def z2(scale):
        return rng.normal(0, scale, (n_mc, 2))

    rows = []
    true_static = np.broadcast_to(nominal, (n_mc, 2, 2)).copy()

    def add_case(name, pred_centres, true_pts=true_static, extra=None):
        for radius in radii:
            prob, margin, cost = margin_and_area(true_pts, pred_centres, radius)
            rows.append({"case": name, "radius_m": radius,
                         "p_containment": prob, "margin_m": margin,
                         "area_cost_m2": cost, **(extra or {})})

    # Static tracking errors displace predicted zone centres, not true points.
    add_case("Unbiased isotropic sigma=15 mm (per eye)",
             nominal[None, :, :] + np.stack([z2(sig), z2(sig)], axis=1))
    for b_mm in (10, 20, 30):
        bias = np.array([b_mm / 1000.0, 0.0])
        err = np.stack([z2(sig), z2(sig)], axis=1)
        add_case(f"Common lateral bias {b_mm} mm", nominal[None, :, :] + err + bias)

    aniso_l = np.stack([rng.normal(0, .020, n_mc), rng.normal(0, .008, n_mc)], axis=1)
    aniso_r = np.stack([rng.normal(0, .020, n_mc), rng.normal(0, .008, n_mc)], axis=1)
    add_case("Anisotropic (20, 8) mm, independent eyes",
             nominal[None, :, :] + np.stack([aniso_l, aniso_r], axis=1))

    rho = 0.6
    z0 = rng.normal(0, 1, (n_mc, 2))
    eye_l = sig * (math.sqrt(rho) * z0 + math.sqrt(1-rho) * rng.normal(0, 1, (n_mc, 2)))
    eye_r = sig * (math.sqrt(rho) * z0 + math.sqrt(1-rho) * rng.normal(0, 1, (n_mc, 2)))
    add_case("Correlated rho=0.6 (exact, marginal sigma 15 mm)",
             nominal[None, :, :] + np.stack([eye_l, eye_r], axis=1))

    zcm = rng.normal(0, 1, (n_mc, 2))
    eye_l = sig * (math.sqrt(.5) * zcm + math.sqrt(.5) * rng.normal(0, 1, (n_mc, 2)))
    eye_r = sig * (math.sqrt(.5) * zcm + math.sqrt(.5) * rng.normal(0, 1, (n_mc, 2)))
    add_case("Head-pose common-mode (rho=0.5)",
             nominal[None, :, :] + np.stack([eye_l, eye_r], axis=1))

    # Latency: predicted zones carry static tracking noise; true intercepts
    # move by the yaw/translation displacement during the latency interval.
    def yaw_disp(lo, hi, t_ms):
        psi = rng.uniform(lo, hi, n_mc)
        disp = core.D_W * np.tan(np.deg2rad(psi * t_ms / 1000.0))
        sign = rng.choice([-1.0, 1.0], n_mc)
        return np.stack([sign * disp, np.zeros(n_mc)], axis=1), disp

    def trans_disp(v, t_ms):
        d = v * t_ms / 1000.0
        # The manuscript specifies lateral eye-translation displacement.
        # Randomize only its left/right sign; do not rotate it into the
        # vertical windshield-plane axis.
        sign = rng.choice([-1.0, 1.0], n_mc)
        return np.stack([sign * np.full(n_mc, d), np.zeros(n_mc)], axis=1), np.full(n_mc, d)

    for t_ms in (114.3, 187.3):
        for lo, hi, label in ((1., 8., "1-8 deg/s"), (10., 40., "10-40 deg/s (aggressive)")):
            disp_xy, disp_mag = yaw_disp(lo, hi, t_ms)
            pred_noise = np.stack([z2(sig), z2(sig)], axis=1)
            pred = nominal[None, :, :] + pred_noise
            true_pts = nominal[None, :, :] + disp_xy[:, None, :]
            add_case(f"Vehicle yaw {label}, T={t_ms:.1f} ms", pred, true_pts,
                     {"disp_p50_mm": float(np.median(disp_mag)*1000),
                      "disp_p95_mm": float(np.percentile(disp_mag,95)*1000)})

    for v, t_ms in ((0.1, 114.3), (0.3, 187.3)):
        disp_xy, disp_mag = trans_disp(v, t_ms)
        pred_noise = np.stack([z2(sig), z2(sig)], axis=1)
        pred = nominal[None, :, :] + pred_noise
        true_pts = nominal[None, :, :] + disp_xy[:, None, :]
        add_case(f"Eye translation {v:.1f} m/s, T={t_ms:.1f} ms", pred, true_pts,
                 {"disp_p50_mm": float(np.median(disp_mag)*1000),
                  "disp_p95_mm": float(np.percentile(disp_mag,95)*1000)})
    return pd.DataFrame(rows)


# ------------------------------------------------- ground-truth sensitivity

GT_G_GRID = [0.45, 0.50, 0.55, 0.60, 0.65]   # evaluation glare threshold
GT_A_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]   # evaluation alignment threshold


def groundtruth_sensitivity(seeds):
    """Sensitivity of the crisp/fuzzy comparison to the evaluation ground truth.

    Controllers see only noisy measurements, so re-scoring the SAME
    full-stage attenuation trajectories under shifted ground-truth
    thresholds (base 0.55/0.50, +/-0.05 and +/-0.10) and under a graded
    ground truth w = g_true * a_true (a veiling-luminance-style weight,
    since veiling luminance grows as the source approaches the line of
    sight) isolates the evaluation bias identified in review comment M2.

    Returns (summary_df, comparison_df, diagnostics_df).
    """
    from scipy import stats as st
    gvals, diag = [], []
    for fam in core.FAMILIES:
        a_min, a_max, cross, f1c0, tot, fixedon = 1.0, 0.0, 0, 0, 0, 0
        for seed in seeds[fam]:
            sc = core.generate_scenario(fam, seed)
            g, a = sc["g_meas"], sc["a_meas"]
            fixedon += np.count_nonzero(g >= 0.55)
            f1c0 += np.count_nonzero((g >= 0.55) & (a < 0.50))
            cross += np.count_nonzero(np.diff((a >= 0.50).astype(int)))
            a_min, a_max = min(a_min, a.min()), max(a_max, a.max())
            tot += len(g)
            alphas = {}
            for ctrl in ("fixed", "crisp", "fuzzy"):
                raw = core.controller_raw(ctrl, sc["g_meas"], sc["a_meas"],
                                          sc["c_meas"])
                alpha, _, _ = core.temporal_pipeline(
                    raw, sc["c_meas"], sc["inhibit"].astype(bool))
                alphas[ctrl] = alpha
            for gt_g in GT_G_GRID:
                for gt_a in GT_A_GRID:
                    w = core.warranted_mask(sc["g_true"], sc["a_true"],
                                            gt_g, gt_a)
                    for ctrl, alpha in alphas.items():
                        wa = float(np.mean(alpha[w])) if w.any() else np.nan
                        na = float(np.mean(alpha[~w]))
                        gvals.append({"family": fam, "seed": seed,
                                      "g_thr": gt_g, "a_thr": gt_a,
                                      "controller": ctrl, "wa": wa, "na": na})
            wgt = sc["g_true"] * sc["a_true"]
            for ctrl, alpha in alphas.items():
                gvals.append({"family": fam, "seed": seed, "g_thr": -1,
                              "a_thr": -1, "controller": ctrl,
                              "wa": float(np.sum(wgt * alpha) / np.sum(wgt)),
                              "na": np.nan})
        diag.append({"family": fam,
                     "pct_frames_glare_above_0.55": 100 * fixedon / tot,
                     "pct_frames_fixed_on_crisp_off": 100 * f1c0 / tot,
                     "alignment_crossings_per_seed": cross / len(seeds[fam]),
                     "a_meas_min": a_min, "a_meas_max": a_max})
    df = pd.DataFrame(gvals)

    rows = []
    for (gt_g, gt_a), d in df.groupby(["g_thr", "a_thr"]):
        graded = gt_g == -1
        label = ("graded w=g*a" if graded else f"G*>={gt_g}, A*>={gt_a}")
        med = d.groupby("controller")["wa"].median()
        med_na = d.groupby("controller")["na"].median()
        for ctrl in ("fixed", "crisp", "fuzzy"):
            rows.append({"definition": label, "controller": ctrl,
                         "median_warranted_attenuation": med.get(ctrl, np.nan),
                         "median_nuisance_attenuation": med_na.get(ctrl, np.nan)})
    summary = pd.DataFrame(rows)

    crows = []
    for (gt_g, gt_a), d in df.groupby(["g_thr", "a_thr"]):
        graded = gt_g == -1
        label = ("graded w=g*a" if graded else f"G*>={gt_g}, A*>={gt_a}")
        for fam in core.FAMILIES + ["ALL"]:
            dd = d if fam == "ALL" else d[d.family == fam]
            xa = dd[dd.controller == "fuzzy"].set_index("seed")["wa"]
            ya = dd[dd.controller == "crisp"].set_index("seed")["wa"]
            common = xa.index.intersection(ya.index)
            x, y = xa.loc[common].to_numpy(), ya.loc[common].to_numpy()
            m = ~(np.isnan(x) | np.isnan(y))
            x, y = x[m], y[m]
            rec = {"definition": label, "family": fam, "n": len(x),
                   "metric": "warranted_attenuation"}
            if len(x) >= 10 and not np.all(x == y):
                res = st.wilcoxon(x, y)
                rec.update(p_value=float(res.pvalue),
                           median_diff=float(np.median(x - y)),
                           pct_seeds_fuzzy_better=float(np.mean(x > y) * 100))
            else:
                rec.update(p_value=np.nan, median_diff=np.nan,
                           pct_seeds_fuzzy_better=np.nan)
            crows.append(rec)
            if not graded:
                xn = dd[dd.controller == "fuzzy"].set_index("seed")["na"]
                yn = dd[dd.controller == "crisp"].set_index("seed")["na"]
                common = xn.index.intersection(yn.index)
                x2, y2 = xn.loc[common].to_numpy(), yn.loc[common].to_numpy()
                m2 = ~(np.isnan(x2) | np.isnan(y2))
                x2, y2 = x2[m2], y2[m2]
                recn = {"definition": label, "family": fam, "n": len(x2),
                        "metric": "nuisance_attenuation"}
                if len(x2) >= 10 and not np.all(x2 == y2):
                    res = st.wilcoxon(x2, y2)
                    recn.update(p_value=float(res.pvalue),
                                median_diff=float(np.median(x2 - y2)),
                                pct_seeds_fuzzy_better=float(np.mean(x2 < y2) * 100))
                else:
                    recn.update(p_value=np.nan, median_diff=np.nan,
                                pct_seeds_fuzzy_better=np.nan)
                crows.append(recn)
    comparison = pd.DataFrame(crows)
    return summary, comparison, pd.DataFrame(diag)


def latency_checks():
    nom = [18.5, 12.0, 22.4, 8.6, 16.8, 36.0]
    stress = [32.0, 25.0, 45.0, 15.0, 25.3, 45.0]
    tau = 0.12
    rows = [
        {"quantity": "Sensing subtotal nominal (ms)", "value": sum(nom[:2]) + nom[3]},
        {"quantity": "Computation subtotal nominal (ms)", "value": nom[2] + nom[4]},
        {"quantity": "Glazing response nominal (ms)", "value": nom[5]},
        {"quantity": "Total nominal (ms)", "value": sum(nom)},
        {"quantity": "Sensing subtotal stressed (ms)", "value": sum(stress[:2]) + stress[3]},
        {"quantity": "Computation subtotal stressed (ms)", "value": stress[2] + stress[4]},
        {"quantity": "Glazing response stressed (ms)", "value": stress[5]},
        {"quantity": "Total stressed (ms)", "value": sum(stress)},
        {"quantity": "Filter 90% rise time tau=0.12 s (ms)", "value": 1000 * tau * math.log(10)},
        {"quantity": "Filter 63% rise time tau=0.12 s (ms)", "value": 1000 * tau},
        {"quantity": "Filter 90% rise time tau=0.05 s (ms)", "value": 1000 * 0.05 * math.log(10)},
        {"quantity": "Filter 90% rise time tau=1.0 s (ms)", "value": 1000 * 1.0 * math.log(10)},
    ]
    return pd.DataFrame(rows)


def main():
    seeds = seed_list()
    with open(OUT / "seed_list.json", "w") as f:
        json.dump(seeds, f, indent=1)

    df, traces = run_main_study(seeds)
    df.to_csv(OUT / "per_seed_metrics.csv", index=False, float_format="%.9f")

    # example traces for figures
    ex = {f"{sc['family']}_{sc['seed']}": sc for sc in traces}
    np.savez(OUT / "example_traces.npz", **{k: np.column_stack(
        [v["time_s"], v["g_meas"], v["a_meas"], v["c_meas"], v["warranted"],
         v["inhibit"]]) for k, v in ex.items()})

    summ, comp = run_statistics(df)
    summ.to_csv(OUT / "summary_statistics.csv", index=False, float_format="%.9f")
    comp.to_csv(OUT / "paired_comparisons.csv", index=False, float_format="%.9f")

    trade = run_tradeoff(seeds)
    trade.to_csv(OUT / "tradeoff_sweep.csv", index=False, float_format="%.9f")

    cont = containment_monte_carlo()
    cont.to_csv(OUT / "containment_montecarlo.csv", index=False, float_format="%.9f")

    gt_sum, gt_cmp, diag = groundtruth_sensitivity(seeds)
    gt_sum.to_csv(OUT / "groundtruth_sensitivity.csv", index=False, float_format="%.9f")
    gt_cmp.to_csv(OUT / "groundtruth_comparison.csv", index=False, float_format="%.9f")
    diag.to_csv(OUT / "alignment_diagnostics.csv", index=False, float_format="%.9f")

    lat = latency_checks()
    lat.to_csv(OUT / "latency_checks.csv", index=False, float_format="%.9f")

    rules = pd.DataFrame(core.RULES)
    rules.to_csv(OUT / "rule_base.csv", index=False)

    with open(OUT / "environment.json", "w") as f:
        import platform
        import matplotlib
        import scipy
        json.dump({"python": sys.version, "platform": platform.platform(),
                   "numpy": np.__version__, "pandas": pd.__version__,
                   "matplotlib": matplotlib.__version__,
                   "scipy": scipy.__version__}, f, indent=2)

    print("Main study:", df.shape, "rows")
    print(summ[summ.family == "ALL"].to_string(index=False))
    print("\nFuzzy vs crisp (ALL families):")
    print(comp[comp.family == "ALL"].to_string(index=False))
    print("\nContainment:")
    print(cont.to_string(index=False))
    print("\nGround-truth sensitivity (crisp vs fuzzy, ALL families):")
    print(gt_cmp[gt_cmp.family == "ALL"].to_string(index=False))
    print("\nAlignment diagnostics:")
    print(diag.to_string(index=False))
    print("\nLatency:")
    print(lat.to_string(index=False))


if __name__ == "__main__":
    main()
