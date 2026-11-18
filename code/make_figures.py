#!/usr/bin/env python3
"""Publication figures for the gaze-aware windshield manuscript."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Arc, Wedge)
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT.parent / "outputs"
FIGDIR = ROOT.parent / "figures"
FIGDIR.mkdir(exist_ok=True)
import core  # noqa: E402

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4,
    "font.family": "DejaVu Sans",
})


def save(fig, stem):
    fig.savefig(FIGDIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGDIR / f"{stem}.eps", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", stem)


# ---------------------------------------------------------------- Fig. 1 geometry
def fig_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1))

    # (a) top view
    ax = axes[0]
    ax.set_aspect("equal")
    D = core.D_W
    ax.plot([D, D], [-0.45, 0.45], color="#1a1a1a", lw=3.0)   # windshield
    ax.text(D + 0.02, 0.40, "windshield", rotation=90, fontsize=8)
    eL, eR = np.array([0, -core.EYE_HALF]), np.array([0, core.EYE_HALF])
    ax.add_patch(Circle(eL, 0.016, fc="#0072B2", ec="none"))
    ax.add_patch(Circle(eR, 0.016, fc="#0072B2", ec="none"))
    ax.text(-0.02, -core.EYE_HALF, "$e_L$", ha="right", va="center")
    ax.text(-0.02, core.EYE_HALF, "$e_R$", ha="right", va="center")
    # eye-to-glare-source ray and its windshield intercept (zone centre)
    src_dir = np.array([np.cos(np.deg2rad(22)), np.sin(np.deg2rad(22))])
    lam = D / src_dir[0]
    P = eR + lam * src_dir
    ax.plot([eR[0], P[0]], [eR[1], P[1]], color="#D55E00", lw=1.3)
    ax.plot(*P, marker="*", ms=13, color="#D55E00")
    ax.text(P[0] - 0.03, P[1] + 0.115, "glare\nsource", color="#D55E00", ha="right", va="bottom")
    # gaze direction (used only for alignment A, not for zone placement)
    gaze_dir = np.array([np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10))])
    Pg = eR + (D / gaze_dir[0]) * gaze_dir
    ax.plot([eR[0], Pg[0]], [eR[1], Pg[1]], color="#0072B2", lw=1.3, ls="--")
    ax.text(0.42, 0.045, "gaze $d_R$", color="#0072B2", ha="left", va="top")
    # attenuation zones centred on the eye-to-source intercepts
    for e in (eL, eR):
        q = e + (D / src_dir[0]) * src_dir
        ax.add_patch(Circle(q, 0.06, fc="#009E73", alpha=0.15, ec="#009E73", lw=1.2))
        ax.plot(*q, marker="o", ms=4, color="#009E73")
    ax.annotate("attenuation zones\n(on eye–source ray)", xy=(0.715, 0.345), xytext=(0.14, 0.315),
                color="#1b6e52", fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#1b6e52", lw=0.9))
    # angle arc
    a1 = np.deg2rad(10); a2 = np.deg2rad(22)
    arc = Arc(eR, 0.55, 0.55, angle=0, theta1=np.degrees(a1), theta2=np.degrees(a2))
    ax.add_patch(arc)
    ax.text(0.17, 0.075, r"$\theta_R$", fontsize=9)
    ax.annotate("", xy=(0.44, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="#555555"))
    ax.text(0.22, -0.03, "$D_W$ = 0.75 m", fontsize=8, color="#555555")
    ax.set_xlim(-0.12, 0.95); ax.set_ylim(-0.30, 0.52)
    ax.set_xlabel("forward distance (m)")
    ax.set_ylabel("lateral position (m)")
    ax.set_title("(a) Top view", fontsize=9)

    # (b) windshield plane
    ax = axes[1]
    ax.set_aspect("equal")
    ax.add_patch(Circle((-core.EYE_HALF, 0), 0.06, fc="#009E73", alpha=0.25, ec="#009E73"))
    ax.add_patch(Circle((core.EYE_HALF, 0), 0.06, fc="#009E73", alpha=0.25, ec="#009E73"))
    ax.add_patch(Circle((-core.EYE_HALF, 0), 0.06 + 0.018, fc="none", ec="#009E73", ls="--", lw=1.0))
    ax.add_patch(Circle((core.EYE_HALF, 0), 0.06 + 0.018, fc="none", ec="#009E73", ls="--", lw=1.0))
    ax.plot([-core.EYE_HALF, core.EYE_HALF], [0, 0], color="#555555", lw=1.0)
    ax.text(0, -0.015, "$s$ = 65 mm", ha="center", va="top", fontsize=8, color="#555555")
    ax.text(-core.EYE_HALF, 0.085, "$r$", ha="center", fontsize=9)
    ax.text(core.EYE_HALF + 0.075, 0.03, "confidence-\ndependent\nmargin", fontsize=7.5, color="#1b6e52")
    rng = np.random.default_rng(3)
    ptsL = np.array([-core.EYE_HALF, 0]) + rng.normal(0, 0.012, (60, 2))
    ptsR = np.array([core.EYE_HALF, 0]) + rng.normal(0, 0.012, (60, 2))
    ax.scatter(ptsL[:, 0], ptsL[:, 1], s=4, color="#0072B2", alpha=0.5)
    ax.scatter(ptsR[:, 0], ptsR[:, 1], s=4, color="#D55E00", alpha=0.5)
    ax.text(-0.16, -0.09, "eye-to-source\nintercepts\nunder error", fontsize=7.5, color="#333333")
    ax.set_xlim(-0.22, 0.28); ax.set_ylim(-0.20, 0.22)
    ax.set_xlabel("lateral position (m)")
    ax.set_ylabel("vertical position (m)")
    ax.set_title("(b) Windshield plane", fontsize=9)
    fig.tight_layout()
    save(fig, "figure_1_geometry")


# --------------------------------------------------------------- Fig. 2 architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.9, 3.0))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.4); ax.axis("off")
    boxes = [
        (0.15, 1.7, 1.75, 1.0, "Driver-facing\ncamera and\nglare sensing"),
        (2.15, 1.7, 1.75, 1.0, "Gaze and source\ngeometry,\nalignment $A$"),
        (4.15, 1.7, 1.75, 1.0, "Crisp or fuzzy\nattenuation\nsupervisor"),
        (6.15, 1.7, 1.75, 1.0, "Confidence gate,\nlow-pass filter,\nhysteresis"),
        (8.35, 1.7, 2.10, 1.0, "Dual-eye localized\nglazing command\n(radius $r(C)$)"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    lw=1.2, edgecolor="#334155", facecolor="#EAF2F8"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
    for i in range(len(boxes) - 1):
        x, y, w, h, _ = boxes[i]
        nx = boxes[i + 1][0]
        ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (nx, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#334155"))
    ax.add_patch(FancyBboxPatch((2.6, 0.15), 5.2, 0.85,
                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                lw=1.1, edgecolor="#9C4221", facecolor="#FFF4E6"))
    ax.text(5.2, 0.58, "Inhibition: invalid tracking or safety event $\\rightarrow$\nimmediate transparent command (fail-transparent)",
            ha="center", va="center", fontsize=8, color="#7C2D12")
    ax.add_patch(FancyArrowPatch((5.2, 1.0), (7.0, 1.7), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, linestyle="--", color="#9C4221"))
    ax.text(5.2, 3.05, "$(G, A, C)$ normalized supervisor inputs", ha="center", fontsize=8, color="#444444")
    fig.tight_layout()
    save(fig, "figure_3_architecture")


# --------------------------------------------------------------- Fig. 3 memberships
def fig_memberships():
    x = np.linspace(0, 1, 501)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4), sharey=True)
    styles = [("#0072B2", "-"), ("#D55E00", "--"), ("#009E73", "-.")]
    for ax, name, xlabel in zip(axes, ("G", "A", "C"),
                                ("Normalized glare $G$",
                                 "Gaze–glare alignment $A$",
                                 "Tracking confidence $C$")):
        for (label, points), (color, ls) in zip(core.MEMBERSHIPS[name].items(), styles):
            ax.plot(x, core.trapmf(x, points), label=label.capitalize(),
                    color=color, linestyle=ls, lw=1.6)
        ax.set_xlabel(xlabel); ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Membership degree")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(0.03, 0.95, label, transform=ax.transAxes, fontweight="bold", va="top")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, "figure_2_memberships")


# --------------------------------------------------------------- Fig. 4 multi-scenario
def fig_multiscenario():
    metrics = pd.read_csv(OUT / "per_seed_metrics.csv")
    full = metrics[metrics.stage == "full"]
    # representative S1 and S5 traces
    sc1 = core.generate_scenario("S1", core.FAMILY_BASE_SEED)
    sc5 = core.generate_scenario("S5", core.FAMILY_BASE_SEED + 4000)
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.6))
    colors = {"fixed": "#D55E00", "crisp": "#0072B2", "fuzzy": "#009E73"}
    names = {"fixed": "Fixed threshold", "crisp": "Crisp gated",
             "fuzzy": "Fuzzy TS"}

    for ax, sc, ttl in ((axes[0, 0], sc1, "(a) S1 clear warranted glare"),
                        (axes[0, 1], sc5, "(b) S5 tracking degradation")):
        t = sc["time_s"]
        # warranted intervals: opaque light-grey bands drawn BEHIND the traces
        # (no transparency, so the EPS renders identically to the PNG)
        w = sc["warranted"].astype(int)
        edges = np.diff(np.concatenate([[0], w, [0]]))
        for s0, s1 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(t[s0], t[min(s1, len(t) - 1)], color="#E6E6E6", lw=0, zorder=0)
        for ctrl in ("fixed", "crisp", "fuzzy"):
            raw = core.controller_raw(ctrl, sc["g_meas"], sc["a_meas"], sc["c_meas"])
            alpha, _, _ = core.temporal_pipeline(raw, sc["c_meas"],
                                                 sc["inhibit"].astype(bool))
            ax.plot(t, alpha, color=colors[ctrl], lw=1.0, label=names[ctrl], zorder=2)
        ax.set_ylim(-0.03, 1.12); ax.set_xlim(0, 60)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"attenuation $\alpha$")
        ax.set_title(ttl, fontsize=9)
    from matplotlib.patches import Patch
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h + [Patch(fc="#E6E6E6", ec="none")], l + ["warranted interval"],
               frameon=False, fontsize=7.5, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.02))

    # (c) per-family warranted attenuation
    ax = axes[1, 0]
    fams = core.FAMILIES
    width = 0.26
    for i, ctrl in enumerate(("fixed", "crisp", "fuzzy")):
        vals = [full[(full.family == f) & (full.controller == ctrl)]
                ["warranted_attenuation"].median() for f in fams]
        ax.bar(np.arange(6) + (i - 1) * width, vals, width,
               color=colors[ctrl], label=names[ctrl])
    ax.set_xticks(range(6)); ax.set_xticklabels(fams)
    ax.set_ylim(0, 1.05); ax.set_ylabel("median warranted attenuation")
    ax.set_xlabel("scenario family")   # colours explained by the figure legend at the top
    ax.set_title("(c) Warranted attenuation", fontsize=9)

    ax = axes[1, 1]
    for i, ctrl in enumerate(("fixed", "crisp", "fuzzy")):
        vals = [full[(full.family == f) & (full.controller == ctrl)]
                ["nuisance_attenuation"].median() for f in fams]
        ax.bar(np.arange(6) + (i - 1) * width, vals, width,
               color=colors[ctrl], label=names[ctrl])
    ax.set_xticks(range(6)); ax.set_xticklabels(fams)
    ax.set_ylabel("median nuisance attenuation")
    ax.set_xlabel("scenario family"); ax.set_ylim(0, 0.18)
    ax.set_title("(d) Nuisance attenuation", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "figure_4_multiscenario")


# --------------------------------------------------------------- Fig. 5 trade-off
def fig_tradeoff():
    tr = pd.read_csv(OUT / "tradeoff_sweep.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.4))
    colors = {"crisp": "#0072B2", "fuzzy": "#009E73"}
    names = {"crisp": "Crisp gated", "fuzzy": "Fuzzy TS"}

    for ctrl in ("crisp", "fuzzy"):
        g = tr[tr.controller == ctrl].groupby("tau_s")
        tau = np.array(sorted(tr.tau_s.unique()))
        wa = g["warranted_attenuation"].median().reindex(tau)
        na = g["nuisance_attenuation"].median().reindex(tau)
        tv = g["total_variation"].median().reindex(tau)
        sw = g["switch_count"].median().reindex(tau)
        ad = g["activation_delay_s"].median().reindex(tau)
        axes[0, 0].plot(tv, wa, color=colors[ctrl], marker="o", ms=3.5, label=names[ctrl])
        axes[0, 1].plot(tv, na, color=colors[ctrl], marker="o", ms=3.5, label=names[ctrl])
        axes[1, 0].plot(tau * 1000, sw, color=colors[ctrl], marker="o", ms=3.5, label=names[ctrl])
        axes[1, 1].plot(tau * 1000, ad * 1000, color=colors[ctrl], marker="o", ms=3.5, label=names[ctrl])
    axes[0, 0].set_xlabel("total variation (median)")
    axes[0, 0].set_ylabel("warranted attenuation (median)")
    axes[0, 1].set_xlabel("total variation (median)")
    axes[0, 1].set_ylabel("nuisance attenuation (median)")
    axes[1, 0].set_xlabel(r"low-pass time constant $\tau$ (ms)")
    axes[1, 0].set_ylabel("switch count (median)")
    axes[1, 1].set_xlabel(r"low-pass time constant $\tau$ (ms)")
    axes[1, 1].set_ylabel("activation delay (ms)")
    axes[1, 1].set_yscale("log")
    for ax, lbl in zip(axes.ravel(), ("(a)", "(b)", "(c)", "(d)")):
        ax.grid(alpha=0.2)
        ax.text(0.02, 0.95, lbl, transform=ax.transAxes, fontweight="bold", va="top")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "figure_5_tradeoff")


# --------------------------------------------------------------- Fig. 6 containment
def fig_containment():
    """Drawn at the 174 mm print width; values read from containment_montecarlo.csv."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    cont = pd.read_csv(OUT / "containment_montecarlo.csv")
    cont = cont[cont.radius_m == 0.0514].reset_index(drop=True)   # Eq. (6) radius at C = 0.93
    short = {
        "Unbiased isotropic sigma=15 mm (per eye)": "Isotropic\n$\\sigma$ = 15 mm",
        "Common lateral bias 10 mm": "Bias\n10 mm",
        "Common lateral bias 20 mm": "Bias\n20 mm",
        "Common lateral bias 30 mm": "Bias\n30 mm",
        "Anisotropic (20, 8) mm, independent eyes": "Anisotropic\n(20, 8) mm",
        "Correlated rho=0.6 (exact, marginal sigma 15 mm)": "Correlated\n$\\rho$ = 0.6",
        "Head-pose common-mode (rho=0.5)": "Common-mode\n$\\rho$ = 0.5",
        "Vehicle yaw 1-8 deg/s, T=114.3 ms": "Yaw 1–8°/s\n114.3 ms",
        "Vehicle yaw 10-40 deg/s (aggressive), T=114.3 ms": "Yaw 10–40°/s\n114.3 ms",
        "Vehicle yaw 1-8 deg/s, T=187.3 ms": "Yaw 1–8°/s\n187.3 ms",
        "Vehicle yaw 10-40 deg/s (aggressive), T=187.3 ms": "Yaw 10–40°/s\n187.3 ms",
        "Eye translation 0.1 m/s, T=114.3 ms": "Trans. 0.1 m/s\n114.3 ms",
        "Eye translation 0.3 m/s, T=187.3 ms": "Trans. 0.3 m/s\n187.3 ms",
    }
    cont = cont[cont.case.isin(short)].reset_index(drop=True)
    p = cont["p_containment"].to_numpy() * 100
    motion = cont["case"].str.contains("yaw|translation").to_numpy()
    c_static, c_lat = "#4C72B0", "#DD8452"
    fig = plt.figure(figsize=(174 / 25.4, 160 / 25.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.62)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(p))
    ax.bar(x, p, 0.72, color=np.where(motion, c_lat, c_static), edgecolor="black", linewidth=0.5)
    for xi, v in zip(x, p):
        ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(95, ls="--", color="black", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([short[c] for c in cont["case"]], rotation=90)
    ax.set_xlim(-0.6, len(p) - 0.4); ax.set_ylim(0, 112)
    ax.set_ylabel("Containment at r = 51.4 mm (%)")
    ax.set_title("(a)", loc="left", fontweight="bold")
    ax.legend(handles=[Patch(fc=c_static, ec="black", lw=0.5, label="Static tracking error"),
                       Patch(fc=c_lat, ec="black", lw=0.5, label="Latency-induced error"),
                       Line2D([0], [0], ls="--", color="black", lw=0.9, label="95% threshold")],
              loc="lower left", framealpha=0.95)

    sub = cont[cont.p_containment < 0.95].sort_values("margin_m")      # the cases below 95%
    labels = [short[c].replace("\n", "\n") for c in sub["case"]]
    margin = sub["margin_m"].to_numpy() * 1000
    area = sub["area_cost_m2"].to_numpy() * 1e4
    bx = fig.add_subplot(gs[1]); bx2 = bx.twinx()
    xb = np.arange(len(sub)); w = 0.36
    bx.bar(xb - w / 2, margin, w, color="#2E6F9E", edgecolor="black", linewidth=0.5, label="Radius margin (mm)")
    bx2.bar(xb + w / 2, area, w, color="#C44E52", edgecolor="black", linewidth=0.5, hatch="///",
            label="Added union area (cm$^2$)")
    for xi, v in zip(xb, margin):
        bx.text(xi - w / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    for xi, v in zip(xb, area):
        bx2.text(xi + w / 2, v + 6, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    top = max(10.0, float(np.nanmax(margin)) * 1.16)
    bx.set_ylim(0, top); bx2.set_ylim(0, top * float(np.nanmax(area)) / float(np.nanmax(margin)))
    bx.set_ylabel("Radius margin (mm)"); bx2.set_ylabel("Added union area (cm$^2$)")
    bx.set_xticks(xb); bx.set_xticklabels(labels)
    bx.set_title("(b)", loc="left", fontweight="bold")
    h1, l1 = bx.get_legend_handles_labels(); h2, l2 = bx2.get_legend_handles_labels()
    bx.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.95)
    fig.subplots_adjust(left=0.085, right=0.915, top=0.96, bottom=0.07)
    fig.savefig(FIGDIR / "figure_6_containment.png", dpi=600, facecolor="white")
    fig.savefig(FIGDIR / "figure_6_containment.eps", facecolor="white")
    plt.close(fig)
    print("saved figure_6_containment")


if __name__ == "__main__":
    fig_geometry()
    fig_architecture()
    fig_memberships()
    fig_multiscenario()
    fig_tradeoff()
    fig_containment()
