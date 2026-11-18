# Online Resource 1 — Simulation Code and Supplementary Material

Simulation code, parameters, seeds, results, and figures supporting:

> **A Gaze-Aware Localized Windshield Attenuation System for Driver Glare
> Mitigation: Dual-Eye Geometry, Confidence-Aware Supervisory Control, and
> Simulation Evaluation**
> *International Journal of Intelligent Transportation Systems Research* (Springer)

**Authors:** Nandish Dave¹ (ORCID 0009-0006-6314-816X), Nimisha Pathak² (ORCID 0000-0003-4664-1340)
¹ Independent Researcher, San Jose, CA, USA
² Department of Applied Mathematics, The Maharaja Sayajirao University of Baroda, Vadodara, Gujarat, India
**Corresponding author:** Nandish Dave — nandish747@gmail.com

This is a **controller-in-the-loop simulation study**. Nothing in this
archive constitutes real-world, hardware, human-subject, or regulatory
validation.

## Layout

```
package/
├── README.md                  ← this file
├── CITATION.cff               ← citation metadata
├── LICENSE                    ← MIT license for the code
├── parameters.json            ← all constants
├── requirements.txt           ← pinned Python dependencies
├── code/
│   ├── run_analysis.py        ← main pipeline (main study, statistics,
│   │                             trade-off sweep, ground-truth sensitivity,
│   │                             containment, latency/power)
│   ├── make_figures.py        ← regenerates all figures from outputs/
│   └── src/core.py            ← scenario generators, controllers,
│                                temporal pipeline, metrics
├── outputs/                   ← results of the runs reported in the article
│   ├── per_seed_metrics.csv       14,400 runs × 15 metrics (4 stages × 3 controllers)
│   ├── summary_statistics.csv     group summaries (medians, IQR, bootstrap CIs)
│   ├── paired_comparisons.csv     paired Wilcoxon / effect sizes / win rates
│   ├── tradeoff_sweep.csv         τ sweep (10 τ × 2 controllers × 6 families × 60 seeds)
│   ├── groundtruth_sensitivity.csv  attenuation under 25 shifted threshold
│   │                                definitions + graded ground truth
│   ├── groundtruth_comparison.csv   paired fuzzy-vs-crisp statistics per definition
│   ├── alignment_diagnostics.csv    per-family alignment threshold diagnostics
│   ├── containment_montecarlo.csv   Table 3 and Fig. 6: 13 error cases × 2 radii
│   ├── latency_checks.csv           latency budget and filter rise times
│   ├── rule_base.csv                27-rule Takagi–Sugeno rule base
│   ├── seed_list.json               every seed used
│   ├── environment.json             platform and package versions of the run
│   └── example_traces.npz           sample traces (S1–S6, one seed each)
├── figures/                   ← EPS + PNG (600 dpi), numbered as in the article
│                                (Fig. 2 membership functions, Fig. 3 architecture)
└── validate.py                ← integrity checks (required files, row counts, seeds)
```

## Geometry convention

World frame: x lateral (right positive), y up, z forward. The eyes sit on the
driver baseline at z = 0, at (±0.0325, 0, 0) m (interocular separation
0.065 m). Each attenuation zone is centred on the intercept of the
eye-to-glare-source ray with the windshield plane z = D_W = 0.75 m; the gaze
direction enters only the alignment term A, not the zone placement.

## Containment Monte Carlo (Table 3, Fig. 6)

`containment_monte_carlo()` in `run_analysis.py` uses 200,000 samples per case
(seed 99) at zone radii of 51.4 mm and 60 mm:

- The true eye-to-source intercepts are at x = ±32.5 mm in the windshield plane.
- **Static cases:** the predicted zone centres are the true intercepts displaced
  by per-eye Gaussian tracking error (σ = 15 mm per axis unless stated) and,
  where applicable, a common lateral bias of 10, 20 or 30 mm.
- **Latency cases:** the predicted centres carry the same tracking error, while
  the true intercepts move during the latency T, by D_W·tan(ψ̇·T) under vehicle
  yaw (ordinary 1–8°/s, aggressive 10–40°/s, random left/right sign) or by v·T
  under lateral eye translation (0.1 and 0.3 m/s).
- A draw succeeds when **both** true intercepts lie inside the union of the two
  predicted circles. The margin for 95% containment is found on a 0.5 mm grid.

Ordinary yaw of 1–8°/s corresponds to speeds of 15–25 m/s on curve radii of
about 180–900 m (ψ̇ = v/R); the 10–40°/s range and the eye-translation speeds
are illustrative stress tests. Table 3 and Fig. 6 report
`outputs/containment_montecarlo.csv` directly.

## Reproduce

```bash
pip install -r requirements.txt
cd code && python3 run_analysis.py    # regenerates ../outputs/ (deterministic)
python3 make_figures.py               # regenerates ../figures/ from ../outputs/
cd .. && python3 validate.py          # integrity checks
```

All stochastic elements derive from per-(family, replicate) seeds recorded in
`outputs/seed_list.json` (`family_base = 20261027` in `parameters.json`); the
containment Monte Carlo uses its own seed (99). Re-running with the pinned
versions in `requirements.txt` (Python 3.13) reproduces the packaged outputs.

## Quick integrity check

```bash
python3 validate.py
```

Verifies the required files and figure pairs exist, CSV row/column counts match
the packaged runs, and `seed_list.json` matches `parameters.json`.
