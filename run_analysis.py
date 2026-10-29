#!/usr/bin/env python3
"""Reproduce the synthetic analysis for the gaze-aware windshield manuscript.

All outputs are synthetic controller-in-the-loop calculations. No human participant,
vehicle, glazing hardware, or photometric experiment is represented by this script.
"""
from pathlib import Path
import json
import math
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

PARAMETERS = {
    "study_status": "Synthetic controller evaluation; not human, vehicle, or hardware validation",
    "random_seed": 20261027,
    "sample_rate_hz": 50.0,
    "duration_s": 60.0,
    "windshield_area_m2": 1.12,
    "projected_interocular_separation_m": 0.065,
    "radius_model": "r = clip(0.05 + 0.02 * (1 - C), 0.05, 0.07) metres",
    "glass_power_density_w_m2": 3.0,
    "fixed_glare_threshold": 0.55,
    "alignment_threshold": 0.55,
    "confidence_gate": 0.65,
    "gate_sensitivity_values": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "low_pass_time_constant_s": 0.12,
    "hysteresis_on": 0.12,
    "hysteresis_off": 0.04,
    "warranted_intervals_s": [[5, 11], [18, 24], [31, 37], [45, 52]],
    "nuisance_glare_intervals_s": [[12, 15], [26, 29], [40, 43]],
    "confidence_dip_intervals_s": [[8, 9], [15.5, 17], [21, 22.2], [33, 35], [41, 42], [49, 50], [53, 54]],
    "safety_inhibit_intervals_s": [[33.2, 33.5], [53.2, 53.5]],
    "noise_sd": {"glare": 0.025, "alignment": 0.025, "confidence": 0.018},
    "memberships": {
        "G": {"low": [0, 0, 0.25, 0.55], "medium": [0.25, 0.55, 0.55, 0.80], "high": [0.55, 0.80, 1, 1]},
        "A": {"low": [0, 0, 0.20, 0.45], "medium": [0.20, 0.45, 0.55, 0.80], "high": [0.55, 0.80, 1, 1]},
        "C": {"low": [0, 0, 0.35, 0.60], "medium": [0.35, 0.60, 0.65, 0.85], "high": [0.65, 0.85, 1, 1]}
    },
    "rule_consequent_formula": "z(i,j,k) = (i*j/4)*(0.8 + 0.1*k), indices low=0, medium=1, high=2",
    "latency_budget_ms": {
        "component": ["Camera exposure", "Image transfer", "Gaze estimation", "Glare/scene estimation", "Control and filtering", "Glazing response"],
        "nominal": [18.5, 12.0, 22.4, 8.6, 16.8, 36.0],
        "stressed": [32.0, 25.0, 45.0, 15.0, 25.3, 45.0]
    }
}

with open(ROOT / "parameters.json", "w", encoding="utf-8") as f:
    json.dump(PARAMETERS, f, indent=2)


def pulse(t, intervals, ramp=0.35):
    """Smooth 0--1 pulse with raised-cosine ramps; overlapping pulses use maximum."""
    y = np.zeros_like(t, dtype=float)
    for start, end in intervals:
        start, end = float(start), float(end)
        cur = np.zeros_like(t, dtype=float)
        plateau = (t >= start + ramp) & (t <= end - ramp)
        cur[plateau] = 1.0
        rise = (t >= start) & (t < start + ramp)
        u = (t[rise] - start) / ramp
        cur[rise] = 0.5 - 0.5 * np.cos(np.pi * u)
        fall = (t > end - ramp) & (t <= end)
        u = (end - t[fall]) / ramp
        cur[fall] = 0.5 - 0.5 * np.cos(np.pi * u)
        y = np.maximum(y, cur)
    return y


def interval_mask(t, intervals):
    mask = np.zeros_like(t, dtype=bool)
    for start, end in intervals:
        mask |= (t >= start) & (t < end)
    return mask


def trapmf(x, points):
    """Trapezoidal/triangular membership, including shoulder shapes."""
    x = np.asarray(x, dtype=float)
    a, b, c, d = [float(v) for v in points]
    y = np.zeros_like(x)
    if b == a:
        y[x >= a] = 1.0
    else:
        rising = (x >= a) & (x < b)
        y[rising] = (x[rising] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1.0
    if d > c:
        falling = (x > c) & (x <= d)
        y[falling] = (d - x[falling]) / (d - c)
    y[x > d] = 0.0
    y[x < a] = 0.0
    return np.clip(y, 0.0, 1.0)


def rule_table():
    rows = []
    labels = ["low", "medium", "high"]
    for i, gl in enumerate(labels):
        for j, al in enumerate(labels):
            for k, cl in enumerate(labels):
                z = (i * j / 4.0) * (0.8 + 0.10 * k)
                rows.append({"rule": len(rows) + 1, "G": gl, "A": al, "C": cl,
                             "i": i, "j": j, "k": k, "singleton_z": round(z, 2)})
    return pd.DataFrame(rows)


RULES = rule_table()
RULES.to_csv(OUT / "rule_base.csv", index=False)


def fuzzy_command(G, A, C):
    """Zero-order Takagi--Sugeno inference with product t-norm and weighted sum."""
    m = {}
    for name, x in (("G", G), ("A", A), ("C", C)):
        for label, points in PARAMETERS["memberships"][name].items():
            m[(name, label)] = trapmf(x, points)
    numerator = np.zeros_like(G, dtype=float)
    denominator = np.zeros_like(G, dtype=float)
    for row in RULES.itertuples(index=False):
        w = m[("G", row.G)] * m[("A", row.A)] * m[("C", row.C)]
        numerator += w * row.singleton_z
        denominator += w
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def union_area(r, s=0.065):
    """Area of union of two equal circles with center separation s."""
    r = np.asarray(r, dtype=float)
    overlap = np.where(s < 2 * r,
                       2 * r * r * np.arccos(np.clip(s / (2 * r), -1, 1))
                       - 0.5 * s * np.sqrt(np.maximum(0, 4 * r * r - s * s)),
                       0.0)
    return 2 * np.pi * r * r - overlap


def temporal_filter(raw, C, inhibit, gamma, tau, theta_on, theta_off, dt):
    beta = 1.0 - math.exp(-dt / tau)
    filtered = np.zeros_like(raw)
    state = np.zeros_like(raw, dtype=int)
    alpha = np.zeros_like(raw)
    for n in range(len(raw)):
        if inhibit[n]:
            filtered[n] = 0.0
            state[n] = 0
            alpha[n] = 0.0
            continue
        target = raw[n] if C[n] >= gamma else 0.0
        previous = filtered[n - 1] if n else 0.0
        filtered[n] = previous + beta * (target - previous)
        previous_state = state[n - 1] if n else 0
        state[n] = previous_state
        if previous_state == 0 and filtered[n] >= theta_on:
            state[n] = 1
        elif previous_state == 1 and filtered[n] <= theta_off:
            state[n] = 0
        alpha[n] = state[n] * filtered[n]
    return alpha, filtered, state


def make_scenario():
    n = int(PARAMETERS["duration_s"] * PARAMETERS["sample_rate_hz"])
    dt = 1.0 / PARAMETERS["sample_rate_hz"]
    t = np.arange(n) * dt
    rng = np.random.default_rng(PARAMETERS["random_seed"])
    warranted = interval_mask(t, PARAMETERS["warranted_intervals_s"]).astype(int)
    nuisance = interval_mask(t, PARAMETERS["nuisance_glare_intervals_s"]).astype(int)
    glare_event = pulse(t, PARAMETERS["warranted_intervals_s"] + PARAMETERS["nuisance_glare_intervals_s"])
    align_event = pulse(t, PARAMETERS["warranted_intervals_s"])
    confidence_dip = pulse(t, PARAMETERS["confidence_dip_intervals_s"], ramp=0.20)
    G = np.clip(0.08 + 0.84 * glare_event + rng.normal(0, PARAMETERS["noise_sd"]["glare"], n), 0, 1)
    A = np.clip(0.12 + 0.80 * align_event + rng.normal(0, PARAMETERS["noise_sd"]["alignment"], n), 0, 1)
    C = np.clip(0.93 - 0.50 * confidence_dip + rng.normal(0, PARAMETERS["noise_sd"]["confidence"], n), 0, 1)
    inhibit = interval_mask(t, PARAMETERS["safety_inhibit_intervals_s"])
    C[inhibit] = 0.0
    radius = np.clip(0.05 + 0.02 * (1 - C), 0.05, 0.07)
    area = union_area(radius, PARAMETERS["projected_interocular_separation_m"])
    return pd.DataFrame({"time_s": t, "G": G, "A": A, "C": C,
                         "warranted": warranted, "nuisance_glare": nuisance,
                         "safety_inhibit": inhibit.astype(int), "radius_m": radius,
                         "dual_eye_union_area_m2": area})


def evaluate(df, name, alpha, active):
    warranted = df["warranted"].to_numpy(dtype=bool)
    nonwarranted = ~warranted
    alpha = np.asarray(alpha, dtype=float)
    active = np.asarray(active, dtype=bool)
    active &= df["safety_inhibit"].to_numpy() == 0
    switches = int(np.count_nonzero(active[1:] != active[:-1]))
    active_areas = df.loc[active, "dual_eye_union_area_m2"].to_numpy()
    time_average_area = float(np.mean(active * df["dual_eye_union_area_m2"].to_numpy()))
    return {
        "controller": name,
        "mean_warranted_attenuation": float(np.mean(alpha[warranted])),
        "false_attenuation_energy": float(np.mean(alpha[nonwarranted])),
        "warranted_error": float(np.mean(np.abs(1.0 - alpha[warranted]))),
        "switch_count": switches,
        "mean_active_union_area_m2": float(np.mean(active_areas)) if active_areas.size else 0.0,
        "p95_active_union_area_m2": float(np.percentile(active_areas, 95)) if active_areas.size else 0.0,
        "time_average_active_area_m2": time_average_area,
        "illustrative_glass_power_w": PARAMETERS["glass_power_density_w_m2"] * time_average_area,
        "active_frames": int(np.count_nonzero(active))
    }


def run_controllers(df, gamma=None):
    if gamma is None:
        gamma = PARAMETERS["confidence_gate"]
    G, A, C = [df[x].to_numpy() for x in ("G", "A", "C")]
    inhibit = df["safety_inhibit"].to_numpy(dtype=bool)
    fixed = ((G >= PARAMETERS["fixed_glare_threshold"]) & ~inhibit).astype(float)
    binary = ((G >= PARAMETERS["fixed_glare_threshold"]) & (C >= gamma) & ~inhibit).astype(float)
    aligned = ((G >= PARAMETERS["fixed_glare_threshold"]) & (A >= PARAMETERS["alignment_threshold"])
               & (C >= gamma) & ~inhibit).astype(float)
    raw = fuzzy_command(G, A, C)
    fuzzy, filtered, hstate = temporal_filter(
        raw, C, inhibit, gamma, PARAMETERS["low_pass_time_constant_s"],
        PARAMETERS["hysteresis_on"], PARAMETERS["hysteresis_off"],
        1.0 / PARAMETERS["sample_rate_hz"])
    outputs = {
        "No dimming": (np.zeros_like(G), np.zeros_like(G, dtype=bool), np.zeros_like(G)),
        "Fixed threshold": (fixed, fixed > 0, fixed),
        "Binary confidence-gated": (binary, binary > 0, binary),
        "Alignment-and-confidence-gated threshold": (aligned, aligned > 0, aligned),
        "Fuzzy low-pass + hysteresis": (fuzzy, hstate > 0, raw)
    }
    return outputs


def save_figure(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.eps", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture_figure():
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.2); ax.axis("off")
    boxes = [
        (0.25, 2.0, 1.65, 1.0, "Driver-facing\nsensing"),
        (2.25, 2.0, 1.65, 1.0, "Gaze and glare\ngeometry"),
        (4.25, 2.0, 1.65, 1.0, "Takagi–Sugeno\nsupervisor"),
        (6.25, 2.0, 1.65, 1.0, "Filter, gate, and\nfail-transparent logic"),
        (8.45, 2.0, 1.65, 1.0, "Localized glazing\ncommand")]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    linewidth=1.4, edgecolor="#334155", facecolor="#EAF2F8"))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8.5)
    for i in range(len(boxes)-1):
        x, y, w, h, _ = boxes[i]
        nx = boxes[i+1][0]
        ax.add_patch(FancyArrowPatch((x+w, y+h/2), (nx, y+h/2), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.2, color="#334155"))
    ax.add_patch(FancyBboxPatch((2.5, 0.25), 5.4, 0.8, boxstyle="round,pad=0.04,rounding_size=0.08",
                                linewidth=1.3, edgecolor="#9C4221", facecolor="#FFF4E6"))
    ax.text(5.2, 0.65, "Safety inhibit or invalid tracking → immediate transparent command",
            ha="center", va="center", fontsize=8.5, color="#7C2D12")
    ax.add_patch(FancyArrowPatch((5.2, 1.05), (7.05, 2.0), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.2, linestyle="--", color="#9C4221"))
    save_figure(fig, "figure_1_system_architecture")


def membership_figure():
    x = np.linspace(0, 1, 501)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    styles = [("#0072B2", "-"), ("#D55E00", "--"), ("#009E73", "-.")]
    for ax, name, xlabel in zip(axes, ("G", "A", "C"), ("Normalized glare, G", "Gaze–glare alignment, A", "Tracking confidence, C")):
        for (label, points), (color, ls) in zip(PARAMETERS["memberships"][name].items(), styles):
            ax.plot(x, trapmf(x, points), label=label.capitalize(), color=color, linestyle=ls, linewidth=1.8)
        ax.set_xlabel(xlabel); ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Membership degree")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(0.02, 0.94, label, transform=ax.transAxes, fontweight="bold", va="top")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure(fig, "figure_2_membership_functions")


def temporal_figure(df):
    outputs = run_controllers(df)
    t = df["time_s"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0, 1.55]})
    for ax in axes:
        for start, end in PARAMETERS["warranted_intervals_s"]:
            ax.axvspan(start, end, color="#DDDDDD", alpha=0.55, lw=0)
    axes[0].plot(t, df["G"], color="#D55E00", lw=1.1, label="Glare G")
    axes[0].plot(t, df["A"], color="#0072B2", lw=1.1, label="Alignment A")
    axes[0].set_ylabel("Normalized value"); axes[0].set_ylim(-0.03, 1.05)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[1].plot(t, df["C"], color="#009E73", lw=1.1)
    axes[1].axhline(PARAMETERS["confidence_gate"], color="#555555", ls="--", lw=1.1, label="Confidence gate")
    axes[1].set_ylabel("Confidence C"); axes[1].set_ylim(-0.03, 1.05)
    axes[1].legend(frameon=False, loc="lower right")
    styles = [("Fixed threshold", "#D55E00", "--"), ("Binary confidence-gated", "#0072B2", "-."),
              ("Fuzzy low-pass + hysteresis", "#009E73", "-")]
    for name, color, ls in styles:
        axes[2].plot(t, outputs[name][0], label=name, color=color, ls=ls, lw=1.25)
    axes[2].set_ylabel("Attenuation α"); axes[2].set_xlabel("Time (s)"); axes[2].set_ylim(-0.03, 1.05)
    axes[2].legend(frameon=False, ncol=3, fontsize=9, loc="upper center")
    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(0.008, 0.91, label, transform=ax.transAxes, fontweight="bold", va="top")
        ax.grid(alpha=0.16)
    fig.text(0.012, 0.012, "Shaded intervals: warranted attenuation", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, "figure_4_temporal_controller_comparison")


def sensitivity_figure(sens):
    x = sens["confidence_gate"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.35))
    axes[0].plot(x, sens["mean_warranted_attenuation"], color="#0072B2", marker="o", lw=1.6, label="Warranted attenuation")
    axes[0].plot(x, sens["warranted_error"], color="#D55E00", marker="s", ls="--", lw=1.6, label="Warranted error")
    axes[0].set_ylabel("Normalized value"); axes[0].set_ylim(0, 1.0)
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].plot(x, sens["false_attenuation_energy"], color="#CC79A7", marker="^", lw=1.6)
    axes[1].set_ylabel("False attenuation energy"); axes[1].set_ylim(0, 0.002)
    axes[2].plot(x, sens["p95_active_union_area_m2"], color="#009E73", marker="D", lw=1.6)
    axes[2].set_ylabel("95th-percentile active area (m²)"); axes[2].set_ylim(0.015, 0.020)
    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.set_xlabel("Confidence gate γ")
        ax.grid(alpha=0.22)
        ax.text(0.02, 0.94, label, transform=ax.transAxes, fontweight="bold", va="top")
    fig.tight_layout()
    save_figure(fig, "figure_5_gate_sensitivity")


def analytical_figure():
    r = np.linspace(0.005, 0.08, 400)
    fraction = 100 * union_area(r, 0.065) / PARAMETERS["windshield_area_m2"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    axes[0].plot(100*r, fraction, color="#0072B2", lw=1.8)
    axes[0].axvline(5, color="#666666", ls="--", lw=1.0)
    axes[0].axvline(7, color="#666666", ls=":", lw=1.0)
    axes[0].set_xlabel("Controller radius (cm)"); axes[0].set_ylabel("Dual-eye union area / windshield (%)")
    axes[0].grid(alpha=0.22)
    margins = np.linspace(0, 3, 400)
    for sigma, color, ls in ((0.02, "#D55E00", "--"), (0.03, "#009E73", "-")):
        p = 100 * (1 - np.exp(-(0.05 + margins*sigma)**2 / (2*sigma*sigma)))
        axes[1].plot(margins, p, color=color, ls=ls, lw=1.8, label=f"σ = {int(sigma*1000)} mm")
    axes[1].set_xlabel("Safety margin m (multiples of σ)")
    axes[1].set_ylabel("Analytical containment probability (%)")
    axes[1].set_ylim(0, 100.5); axes[1].grid(alpha=0.22); axes[1].legend(frameon=False)
    for label, ax in zip(("(a)", "(b)"), axes):
        ax.text(0.02, 0.94, label, transform=ax.transAxes, fontweight="bold", va="top")
    fig.tight_layout()
    save_figure(fig, "figure_3_geometry_and_containment")


def analytical_checks():
    rows = []
    for radius in (0.05, 0.06, 0.07):
        single = math.pi * radius**2
        dual = float(union_area(radius, 0.065))
        rows.extend([
            {"quantity": f"Single circle area, r={radius:.2f} m", "value": single, "unit": "m^2"},
            {"quantity": f"Dual-eye union area, r={radius:.2f} m, s=0.065 m", "value": dual, "unit": "m^2"},
            {"quantity": f"Dual-eye union fraction, r={radius:.2f} m", "value": 100*dual/1.12, "unit": "%"},
            {"quantity": f"Illustrative glass power at 3 W/m^2, r={radius:.2f} m", "value": 3*dual, "unit": "W"}
        ])
    for sigma in (0.02, 0.03):
        for margin in (0, 1, 2, 3):
            k = 0.05 + margin*sigma
            p = 1 - math.exp(-k*k/(2*sigma*sigma))
            rows.append({"quantity": f"Containment, sigma={sigma:.2f} m, margin={margin} sigma", "value": 100*p, "unit": "%"})
    return pd.DataFrame(rows)


def main():
    df = make_scenario()
    outputs = run_controllers(df)
    df["fuzzy_raw_command"] = outputs["Fuzzy low-pass + hysteresis"][2]
    summaries = []
    for name, (alpha, active, raw) in outputs.items():
        safe_name = {"No dimming": "alpha_no_dimming", "Fixed threshold": "alpha_fixed_threshold",
                     "Binary confidence-gated": "alpha_binary_gated",
                     "Alignment-and-confidence-gated threshold": "alpha_aligned_gated",
                     "Fuzzy low-pass + hysteresis": "alpha_fuzzy"}[name]
        df[safe_name] = alpha
        df[safe_name.replace("alpha_", "active_")] = active.astype(int)
        summaries.append(evaluate(df, name, alpha, active))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "controller_summary.csv", index=False, float_format="%.9f")

    sens_rows = []
    for gamma in PARAMETERS["gate_sensitivity_values"]:
        gamma_outputs = run_controllers(df, gamma=gamma)
        alpha, active, _ = gamma_outputs["Fuzzy low-pass + hysteresis"]
        row = evaluate(df, "Fuzzy low-pass + hysteresis", alpha, active)
        row["confidence_gate"] = gamma
        sens_rows.append(row)
    sens = pd.DataFrame(sens_rows)
    sens = sens[["confidence_gate"] + [c for c in sens.columns if c != "confidence_gate"]]
    sens.to_csv(OUT / "gate_sensitivity.csv", index=False, float_format="%.9f")

    df.to_csv(OUT / "synthetic_frame_data.csv", index=False, float_format="%.9f")
    checks = analytical_checks()
    checks.to_csv(OUT / "analytical_checks.csv", index=False, float_format="%.9f")
    latency = pd.DataFrame(PARAMETERS["latency_budget_ms"])
    latency.loc[len(latency)] = ["Total"] + [latency[c].sum() for c in ("nominal", "stressed")]
    latency.to_csv(OUT / "latency_budget.csv", index=False, float_format="%.3f")

    architecture_figure()
    membership_figure()
    temporal_figure(df)
    sensitivity_figure(sens)
    analytical_figure()

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__
    }
    with open(OUT / "environment.json", "w", encoding="utf-8") as f:
        json.dump(environment, f, indent=2)
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"controller_summary": summaries, "gate_sensitivity": sens_rows}, f, indent=2)

    pd.set_option("display.max_columns", None)
    print(summary.to_string(index=False))
    print("\nGate sensitivity:\n", sens[["confidence_gate", "mean_warranted_attenuation", "false_attenuation_energy", "warranted_error", "p95_active_union_area_m2"]].to_string(index=False))
    print("\nLatency totals (ms):", latency.iloc[-1].to_dict())
    print("Outputs written to", OUT)


if __name__ == "__main__":
    main()
