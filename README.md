# Gaze-Aware Windshield Attenuation — Code and Synthetic Data

**Online Resource 1 for:** *Gaze-Aware Windshield Attenuation Using Dual-Eye Geometry and Confidence-Gated Takagi–Sugeno Control: Analytical and Synthetic Evaluation*
**Journal:** International Journal of Computational Intelligence Systems (submitted)
**Authors:** Nandish Dave¹, Nimisha Pathak²
¹ Independent Researcher, San Jose, CA, USA
² Department of Applied Mathematics, The Maharaja Sayajirao University of Baroda, Vadodara, Gujarat, India
**Corresponding author:** Nandish Dave — nandish747@gmail.com
**Archived version (DOI):** 10.5281/zenodo.23757196

---

## What this repository contains

This repository reproduces every analytical value, simulation result, table and figure in the article. It runs a deterministic 60-second synthetic scenario through five controllers, evaluates them with the metrics defined in the paper, and computes the closed-form geometry and eye-error containment results.

> **Scope.** All results are synthetic, controller-in-the-loop calculations. Nothing here represents human participants, a vehicle, glazing hardware, or photometric measurement.

## Files

| File | Purpose |
| --- | --- |
| `run_analysis.py` | Runs the full analysis and writes all outputs |
| `validate_results.py` | Checks the outputs against every number reported in the paper |
| `parameters.json` | Human-readable copy of all parameters (see note below) |
| `requirements.txt` | Pinned Python package versions |
| `outputs/` | Results from a reference run (regenerated each time the script runs) |
| `CITATION.cff` | Citation metadata used by GitHub and Zenodo |
| `LICENSE` | MIT licence |

**Note on `parameters.json`:** this file is *written* by `run_analysis.py` from the `PARAMETERS` dictionary at the top of the script; the script does not read it. To change a parameter, edit `PARAMETERS` in `run_analysis.py` and rerun.

## Requirements

- Python **3.10 or newer** (tested with Python 3.12)
- numpy 2.2.5, pandas 2.3.2, matplotlib 3.10.3 (pinned in `requirements.txt`)

The results were also reproduced, identical to every reported digit, with Python 3.13, numpy 2.5.3, pandas 3.0.5 and matplotlib 3.11.2.

## Quick start

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
python validate_results.py
```

Windows (PowerShell):

```powershell
py -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_analysis.py
python validate_results.py
```

`run_analysis.py` prints the controller summary, the confidence-gate sensitivity table and the latency totals, and writes everything to `outputs/`. It finishes in under a minute on a laptop. Warnings that "The PostScript backend does not support transparency" are expected: the EPS figures render the grey shaded intervals as opaque, which does not affect the plotted data.

`validate_results.py` compares the outputs with the values printed in the manuscript and should end with:

```
ALL CHECKS PASSED
```

It exits with status 0 on success and 1 if any check fails.

## Outputs and where they appear in the paper

| Output file | Paper item |
| --- | --- |
| `outputs/rule_base.csv` | Table 2 — complete 27-rule Takagi–Sugeno rule base |
| `outputs/analytical_checks.csv` | Sect. 3.2, Table 3, Sect. 7.2 — union areas, containment, static power |
| `outputs/controller_summary.csv` | Table 4 and Sect. 6.2 — controller comparison |
| `outputs/gate_sensitivity.csv` | Table 5 — confidence-gate sensitivity |
| `outputs/latency_budget.csv` | Table 6 — latency allocations |
| `outputs/synthetic_frame_data.csv` | All 3,000 frames: inputs, labels, radius, area, and every controller's command and active state |
| `outputs/summary.json` | Controller and sensitivity results in one file |
| `outputs/environment.json` | Python, platform and package versions used for the run |
| `outputs/figures/figure_1_system_architecture.(png\|eps)` | Fig. 1 |
| `outputs/figures/figure_2_membership_functions.(png\|eps)` | Fig. 2 |
| `outputs/figures/figure_3_geometry_and_containment.(png\|eps)` | Fig. 3 |
| `outputs/figures/figure_4_temporal_controller_comparison.(png\|eps)` | Fig. 4 |
| `outputs/figures/figure_5_gate_sensitivity.(png\|eps)` | Fig. 5 |

PNG figures are saved at 600 dpi; EPS versions are vector.

## Method summary

**Scenario.** One deterministic 60 s trace at 50 Hz (3,000 frames), random seed `20261027`. Normalized inputs are glare *G*, gaze–glare alignment *A* and tracking confidence *C*, each built from a baseline value, raised-cosine event pulses and independent Gaussian noise (SD 0.025, 0.025, 0.018).

| Event | Intervals (s) |
| --- | --- |
| Warranted attenuation (high *G* and high *A*) | 5–11, 18–24, 31–37, 45–52 |
| Nuisance glare (high *G*, low *A*) | 12–15, 26–29, 40–43 |
| Confidence dips | 8–9, 15.5–17, 21–22.2, 33–35, 41–42, 49–50, 53–54 |
| Safety inhibit (*C* forced to 0) | 33.2–33.5, 53.2–53.5 |

**Geometry.** Each eye has a circular zone of radius *r* = clip(0.05 + 0.02(1 − *C*), 0.05, 0.07) m. The dual-eye union area uses a projected interocular separation of 0.065 m on a 1.12 m² windshield.

**Controllers.**

| Controller | Rule |
| --- | --- |
| No dimming | α = 0 |
| Fixed threshold | α = 1 if *G* ≥ 0.55 |
| Binary confidence-gated | α = 1 if *G* ≥ 0.55 and *C* ≥ γ |
| Alignment-and-confidence-gated threshold | α = 1 if *G* ≥ 0.55, *A* ≥ 0.55 and *C* ≥ γ |
| Fuzzy low-pass + hysteresis | Zero-order Takagi–Sugeno inference (product t-norm, 27 rules, weighted-sum defuzzification), confidence gate γ, first-order low-pass filter (τ = 0.12 s), hysteresis (on 0.12, off 0.04) |

The nominal gate is γ = 0.65; Table 5 sweeps γ from 0.50 to 0.90. Every controller outputs α = 0 during safety-inhibit frames.

**Membership functions** (trapezoids *a*, *b*, *c*, *d*). For each input, the low, medium and high degrees sum to one (a strong fuzzy partition).

| Input | Low | Medium | High |
| --- | --- | --- | --- |
| *G* | 0, 0, 0.25, 0.55 | 0.25, 0.55, 0.55, 0.80 | 0.55, 0.80, 1, 1 |
| *A* | 0, 0, 0.20, 0.45 | 0.20, 0.45, 0.55, 0.80 | 0.55, 0.80, 1, 1 |
| *C* | 0, 0, 0.35, 0.60 | 0.35, 0.60, 0.65, 0.85 | 0.65, 0.85, 1, 1 |

**Rule consequents.** *z*(*i*, *j*, *k*) = (*i*·*j* / 4)(0.8 + 0.1*k*), with indices low = 0, medium = 1, high = 2 for *G*, *A*, *C*.

**Metrics** (computed in `evaluate()`; *w*ₜ = 1 in warranted frames):

| Metric | Definition |
| --- | --- |
| Mean warranted attenuation | mean of α over warranted frames |
| False-attenuation energy | mean of α over non-warranted frames |
| Warranted error | mean of \|1 − α\| over warranted frames |
| Switch count | number of changes in the binary active state (inhibit frames counted as inactive) |
| Mean / 95th-percentile active area | union area over active frames |
| Illustrative glass power | 3 W/m² × union area averaged over all frames (active frames contribute their area, inactive frames zero) |

**Analytical checks.** Union areas and fractions for *r* = 5, 6, 7 cm; containment probability *P* = 1 − exp[−(*r* + *m*σ)² / (2σ²)] for σ = 20, 30 mm and margins *m* = 0–3; static glass power at 3 W/m².

## Reproducibility notes

- The scenario uses a fixed seed, so every run produces identical numbers.
- Latency values in Table 6 are engineering allocations defined in `PARAMETERS`; the script only sums them.
- Glass power is an illustrative glass-only estimate; it excludes cameras, computation, drivers and conversion losses.

## Use of AI assistance

Large language model assistants were used to help write and check parts of this code and documentation, as disclosed in the article. The authors reviewed and verified all code and outputs.

## Licence

Code is released under the MIT Licence (see `LICENSE`).

## How to cite

If you use this code, please cite the article and this archive:

> Dave, N., Pathak, N.: Code and synthetic data for: Gaze-Aware Windshield Attenuation Using Dual-Eye Geometry and Confidence-Gated Takagi–Sugeno Control. Zenodo (2026). DOI: 10.5281/zenodo.23757196
