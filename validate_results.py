#!/usr/bin/env python3
"""Check that run_analysis.py outputs match the values printed in the manuscript.

Usage (from the folder that contains run_analysis.py):
    python run_analysis.py
    python validate_results.py

Every check prints PASS or FAIL. Paper values are copied from the manuscript
tables; the tolerance is half a unit in the last printed digit.
"""
from pathlib import Path
import sys
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
failures = 0


def check(label, computed, paper, decimals):
    global failures
    ok = round(float(computed), decimals) == round(float(paper), decimals)
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label}: computed {computed:.{decimals + 2}f}, paper {paper}")


# Table 4: controller comparison
s = pd.read_csv(OUT / "controller_summary.csv").set_index("controller")
table4 = {
    "Fixed threshold":             (0.92800, 0.22457, 0.07200, 16, 0.015286, 0.023739),
    "Binary confidence-gated":     (0.76640, 0.20229, 0.23360, 24, 0.014547, 0.019086),
    "Alignment-and-confidence-gated threshold": (0.76560, 0.00000, 0.23440, 16, 0.014548, 0.013922),
    "Fuzzy low-pass + hysteresis": (0.75149, 0.00198, 0.24851, 16, 0.014806, 0.016079),
}
for name, (wa, fe, we, sw, area, pw) in table4.items():
    r = s.loc[name]
    check(f"Table 4 {name} warranted alpha", r.mean_warranted_attenuation, wa, 5)
    check(f"Table 4 {name} false energy", r.false_attenuation_energy, fe, 5)
    check(f"Table 4 {name} warranted error", r.warranted_error, we, 5)
    check(f"Table 4 {name} switches", r.switch_count, sw, 0)
    check(f"Table 4 {name} mean active area", r.mean_active_union_area_m2, area, 6)
    check(f"Table 4 {name} glass power", r.illustrative_glass_power_w, pw, 6)

# Sect. 6.2 text: fuzzy p95 area and p95 power
f = s.loc["Fuzzy low-pass + hysteresis"]
check("Sect. 6.2 fuzzy p95 area", f.p95_active_union_area_m2, 0.018209, 6)
check("Sect. 6.2 fuzzy p95 power", 3 * f.p95_active_union_area_m2, 0.054628, 6)

# Table 5: gate sensitivity
g = pd.read_csv(OUT / "gate_sensitivity.csv").set_index("confidence_gate")
table5 = {
    0.50: (0.76412, 0.00198, 0.23588, 0.019085), 0.55: (0.75830, 0.00198, 0.24170, 0.018931),
    0.60: (0.75438, 0.00198, 0.24562, 0.018558), 0.65: (0.75149, 0.00198, 0.24851, 0.018209),
    0.70: (0.74743, 0.00198, 0.25257, 0.018178), 0.75: (0.74442, 0.00198, 0.25558, 0.017875),
    0.80: (0.74164, 0.00198, 0.25836, 0.017678), 0.85: (0.73557, 0.00198, 0.26443, 0.016797),
    0.90: (0.69048, 0.00186, 0.30952, 0.016610),
}
for gamma, (wa, fe, we, p95) in table5.items():
    r = g.loc[gamma]
    check(f"Table 5 gamma={gamma} warranted alpha", r.mean_warranted_attenuation, wa, 5)
    check(f"Table 5 gamma={gamma} false energy", r.false_attenuation_energy, fe, 5)
    check(f"Table 5 gamma={gamma} warranted error", r.warranted_error, we, 5)
    check(f"Table 5 gamma={gamma} p95 area", r.p95_active_union_area_m2, p95, 6)

# Table 3 and Sect. 3.2 / 7.2: analytical values
a = pd.read_csv(OUT / "analytical_checks.csv").set_index("quantity")["value"]
check("Union area r=5 cm", a["Dual-eye union area, r=0.05 m, s=0.065 m"], 0.013862, 6)
check("Union area r=7 cm", a["Dual-eye union area, r=0.07 m, s=0.065 m"], 0.024155, 6)
check("Union fraction r=5 cm (%)", a["Dual-eye union fraction, r=0.05 m"], 1.2377, 4)
check("Union fraction r=7 cm (%)", a["Dual-eye union fraction, r=0.07 m"], 2.1567, 4)
check("Static glass power r=5 cm", a["Illustrative glass power at 3 W/m^2, r=0.05 m"], 0.041585, 6)
table3 = {(0.02, 0): 95.606307, (0.02, 1): 99.781251, (0.02, 2): 99.995993, (0.02, 3): 99.999973,
          (0.03, 0): 75.064779, (0.03, 1): 97.143450, (0.03, 2): 99.879614, (0.03, 3): 99.998134}
for (sigma, m), p in table3.items():
    check(f"Table 3 sigma={int(sigma * 1000)} mm, m={m}",
          a[f"Containment, sigma={sigma:.2f} m, margin={m} sigma"], p, 6)

# Table 6: latency totals
lat = pd.read_csv(OUT / "latency_budget.csv").set_index("component")
check("Table 6 nominal total", lat.loc["Total", "nominal"], 114.3, 1)
check("Table 6 stressed total", lat.loc["Total", "stressed"], 187.3, 1)

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
