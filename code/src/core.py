#!/usr/bin/env python3
"""Core simulation components for the gaze-aware localized windshield study.

Everything here is synthetic controller-in-the-loop simulation. No human
participant, vehicle, glazing hardware, or photometric experiment is involved.

Geometry convention
-------------------
World frame: x lateral (driver right positive), y vertical (up positive),
z longitudinal (forward positive). The windshield is the plane z = D_W
with D_W = 0.75 m in front of the nominal eye plane. The two eyes sit at
z = 0, separated along the lateral axis x (NOT the forward axis):
left eye e_L = (-S_IO/2, 0, 0), right eye e_R = (+S_IO/2, 0, 0), with
S_IO = 0.065 m the projected interocular separation.

Attenuation-zone centring (eye-to-source ray): each interception zone is
centred at the intersection p_j of the eye-to-glare-source ray
e_j + lambda*s_j (lambda >= 0) with the windshield plane z = D_W,

    p_j = e_j + ((D_W - e_{j,z}) / s_{j,z}) * s_j ,

where s_j is the unit direction from eye j toward the glare source. The
gaze direction d_j is used ONLY in the gaze-glare alignment measure A and
in the supervisory decision; it does not position the zone, because glare
reaches the eye along s_j regardless of where the driver looks.

Directions are given as (azimuth, elevation) in degrees and converted to
unit vectors. For eye j, theta_j = arccos(d_j . s_j) is the angular
separation between the estimated gaze direction d_j and the direction s_j
toward the glare source. Alignment is A_j = exp(-theta_j^2 / (2*THETA_A^2))
with THETA_A = 15 deg, combined as A = max(A_L, A_R).
"""
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants
FS = 50.0                 # sample rate (Hz)
DURATION = 60.0           # scenario duration (s)
N = int(FS * DURATION)
DT = 1.0 / FS
D_W = 0.75                # eye plane -> windshield distance (m)
S_IO = 0.065              # projected interocular separation (m)
EYE_HALF = S_IO / 2.0
THETA_A = np.deg2rad(15.0)          # alignment Gaussian angular scale (rad)
WARRANT_GLARE = 0.55                # fixed glare threshold
WARRANT_ALIGN = 0.50                # true-geometry alignment warrant level
GAMMA = 0.65                        # confidence gate
TAU = 0.12                          # low-pass time constant (s)
HYS_ON, HYS_OFF = 0.12, 0.04        # hysteresis thresholds
GLASS_POWER_DENSITY = 3.0           # W/m^2 illustrative
WINDSHIELD_AREA = 1.12              # m^2
NOISE_SD = {"glare": 0.025, "alignment": 0.025, "confidence": 0.018}

FAMILIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
SEEDS_PER_FAMILY = 200
FAMILY_BASE_SEED = 20261027

# Membership functions (from parameters.json)
MEMBERSHIPS = {
    "G": {"low": [0, 0, 0.25, 0.55], "medium": [0.25, 0.50, 0.75, 0.75],
          "high": [0.55, 0.80, 1, 1]},
    "A": {"low": [0, 0, 0.20, 0.45], "medium": [0.20, 0.50, 0.80, 0.80],
          "high": [0.55, 0.80, 1, 1]},
    "C": {"low": [0, 0, 0.35, 0.60], "medium": [0.35, 0.65, 0.85, 0.85],
          "high": [0.65, 0.85, 1, 1]},
}


def trapmf(x, points):
    x = np.asarray(x, dtype=float)
    a, b, c, d = [float(v) for v in points]
    y = np.zeros_like(x)
    if b == a:
        y[x >= a] = 1.0
    else:
        r = (x >= a) & (x < b)
        y[r] = (x[r] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1.0
    if d > c:
        f = (x > c) & (x <= d)
        y[f] = (d - x[f]) / (d - c)
    y[x > d] = 0.0
    y[x < a] = 0.0
    return np.clip(y, 0.0, 1.0)


def rule_base():
    """Complete 27-rule zero-order Takagi-Sugeno table, z = (i*j/4)*(0.8+0.1k)."""
    rows, labels = [], ["low", "medium", "high"]
    for i, gl in enumerate(labels):
        for j, al in enumerate(labels):
            for k, cl in enumerate(labels):
                rows.append({"rule": len(rows) + 1, "G": gl, "A": al, "C": cl,
                             "i": i, "j": j, "k": k,
                             "z": round((i * j / 4.0) * (0.8 + 0.1 * k), 2)})
    return rows


RULES = rule_base()


def fuzzy_command(G, A, C):
    """Vectorized zero-order TS inference, product t-norm, weighted average."""
    m = {}
    for name, x in (("G", G), ("A", A), ("C", C)):
        for label, pts in MEMBERSHIPS[name].items():
            m[(name, label)] = trapmf(x, pts)
    num = np.zeros_like(G, dtype=float)
    den = np.zeros_like(G, dtype=float)
    for r in RULES:
        w = m[("G", r["G"])] * m[("A", r["A"])] * m[("C", r["C"])]
        num += w * r["z"]
        den += w
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)


def union_area(r, s=S_IO):
    """Area of union of two equal circles of radius r separated by s."""
    r = np.asarray(r, dtype=float)
    overlap = np.where(s < 2 * r,
                       2 * r * r * np.arccos(np.clip(s / (2 * r), -1, 1))
                       - 0.5 * s * np.sqrt(np.maximum(0, 4 * r * r - s * s)),
                       0.0)
    return 2 * np.pi * r * r - overlap


def zone_intercept(eye, src_dir, d_w=D_W):
    """Intersection of the eye-to-source ray with the windshield plane.

    eye: (..., 3) eye position; src_dir: (..., 3) unit source direction.
    Returns p = eye + lambda * src_dir with the plane z = d_w, i.e.
    lambda = (d_w - eye[..., 2]) / src_dir[..., 2]. This is the centre of
    the attenuation zone (see core.py module docstring); the gaze direction
    must NOT be used here.
    """
    eye = np.asarray(eye, dtype=float)
    src_dir = np.asarray(src_dir, dtype=float)
    lam = (d_w - eye[..., 2]) / src_dir[..., 2]
    return eye + lam[..., None] * src_dir


def warranted_mask(g_true, a_true, g_thr, a_thr):
    """Ground-truth 'warranted' mask under arbitrary evaluation thresholds.

    The controllers never see this mask; it is used only for scoring, which
    is what makes ground-truth threshold sensitivity analysis possible.
    """
    return (g_true >= g_thr) & (a_true >= a_thr)


def dir_vec(az_deg, el_deg):
    """(azimuth, elevation) degrees -> unit vector (x right, y up, z forward)."""
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    return np.stack([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)],
                    axis=-1)


def angular_sep(d, s):
    """Angular separation (rad) between unit-vector directions d and s."""
    c = np.clip(np.sum(d * s, axis=-1), -1.0, 1.0)
    return np.arccos(c)


def alignment_from_dirs(gaze_az, gaze_el, src_az, src_el, rng, noise_sd=None):
    """Compute dual-eye max alignment A from gaze/source directions.

    Left/right eye directions differ by a small vergence angle so that the
    two eyes do not produce identical alignment values.
    """
    src = dir_vec(src_az, src_el)
    out = np.empty(np.broadcast(np.asarray(gaze_az), np.asarray(gaze_el)).shape)
    az = np.asarray(gaze_az, dtype=float)
    el = np.asarray(gaze_el, dtype=float)
    best = np.full(az.shape, -np.inf)
    for sign in (-1.0, 1.0):
        d = dir_vec(az + sign * 0.8, el - 0.4)     # small vergence offset
        th = angular_sep(d, src)
        a = np.exp(-th ** 2 / (2 * THETA_A ** 2))
        best = np.maximum(best, a)
    if noise_sd:
        best = best + rng.normal(0, noise_sd, best.shape)
    return np.clip(best, 0.0, 1.0)


# ------------------------------------------------------------ scenario model

def _rand_events(rng, n_events, t_min, t_max, dur_min, dur_max, min_gap=3.0):
    """Non-overlapping random events on [t_min, t_max]."""
    events = []
    tries = 0
    while len(events) < n_events and tries < 400:
        tries += 1
        d = rng.uniform(dur_min, dur_max)
        s = rng.uniform(t_min, t_max - d)
        if all(abs(s - a) > min_gap + d and abs(s - a) > min_gap + b
               for a, b in events):
            events.append((s, s + d))
    return sorted(events)


def _smooth_mask(t, events, ramp=0.35):
    y = np.zeros_like(t)
    for start, end in events:
        cur = np.zeros_like(t)
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


def _interval_mask(t, events):
    m = np.zeros_like(t, dtype=bool)
    for s, e in events:
        m |= (t >= s) & (t < e)
    return m


def generate_scenario(family, seed):
    """Generate one synthetic scenario trace with true and measured signals.

    Returns a dict of 1-D arrays of length N plus the seed and family.
    True geometry fields (starred) are hidden from all controllers and used
    only for the warranted ground-truth mask and for analysis.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N) * DT

    if family == "S1":        # clear warranted glare
        n_ev = rng.integers(3, 6)
        ev = _rand_events(rng, n_ev, 3, 55, 4, 8)
        src_az = rng.uniform(-12, 12)
        src_el = rng.uniform(6, 18)
        g_mag = rng.uniform(0.75, 0.95)
        gaze_az = src_az + rng.normal(0, 2.0, N)
        gaze_el = src_el + rng.normal(0, 2.0, N)
        g_true = np.clip(0.10 + (g_mag - 0.10) * _smooth_mask(t, ev), 0, 1)
        c_true = np.clip(0.93 + rng.normal(0, 0.02, N)
                         - 0.45 * _smooth_mask(t, _rand_events(rng, 1, 5, 50, 1, 2)),
                         0, 1)
        inhibit = _rand_events(rng, 1, 3, 55, 0.3, 0.6)
        eye_bias_mm = 0.0
    elif family == "S2":      # nuisance glare (driver not facing the source)
        n_ev = rng.integers(2, 5)
        ev = _rand_events(rng, n_ev, 3, 55, 4, 8)
        src_az = rng.uniform(-15, 15)
        src_el = rng.uniform(8, 20)
        g_mag = rng.uniform(0.75, 0.95)
        off = rng.uniform(32, 45) * rng.choice([-1, 1])
        gaze_az = off + rng.normal(0, 3.0, N)
        gaze_el = rng.uniform(-8, 2) + rng.normal(0, 2.0, N)
        g_true = np.clip(0.10 + (g_mag - 0.10) * _smooth_mask(t, ev), 0, 1)
        c_true = np.clip(0.93 + rng.normal(0, 0.02, N), 0, 1)
        inhibit = _rand_events(rng, 1, 3, 55, 0.3, 0.6)
        eye_bias_mm = 0.0
    elif family == "S3":      # near-threshold oscillation
        ev = [(2.0, 58.0)]
        p1, p2, p3 = (rng.uniform(4, 9), rng.uniform(4, 9), rng.uniform(5, 10))
        ph1, ph2, ph3 = rng.uniform(0, 2 * np.pi, 3)
        g_true = np.clip(0.55 + 0.09 * np.sin(2 * np.pi * t / p1 + ph1)
                         + rng.normal(0, 0.015, N), 0, 1)
        src_az = 8.0 * np.sin(2 * np.pi * t / p2 + ph2)
        src_el = 10 + 4.0 * np.sin(2 * np.pi * t / p1 + ph1)
        gaze_az = src_az + 12.0 * np.sin(2 * np.pi * t / p3 + ph3)
        gaze_el = src_el - 8.0 * np.sin(2 * np.pi * t / p2 + ph2)
        c_true = np.clip(0.65 + 0.12 * np.sin(2 * np.pi * t / p3 + ph3)
                         + rng.normal(0, 0.02, N), 0, 1)
        inhibit = _rand_events(rng, 1, 10, 50, 0.3, 0.5)
        eye_bias_mm = 0.0
    elif family == "S4":      # slow drift (driving-related direction changes)
        ev = [(2.0, 58.0)]
        p = rng.uniform(20, 40)
        ph = rng.uniform(0, 2 * np.pi)
        src_az = 14.0 * np.sin(2 * np.pi * t / p + ph)
        src_el = 11 + 6.0 * np.sin(2 * np.pi * t / (p * 0.7) + ph / 2)
        lag = rng.integers(5, 30)          # frames of gaze lag on the source
        gaze_az = np.roll(src_az, lag) + rng.normal(0, 1.5, N)
        gaze_el = np.roll(src_el, lag) + rng.normal(0, 1.5, N)
        crossing = np.exp(-(src_az ** 2 + (src_el - 11) ** 2) / (2 * 5.0 ** 2))
        g_true = np.clip(0.12 + 0.72 * crossing, 0, 1)
        c_true = np.clip(0.90 + rng.normal(0, 0.02, N), 0, 1)
        inhibit = _rand_events(rng, 1, 10, 50, 0.3, 0.5)
        eye_bias_mm = 0.0
    elif family == "S5":      # tracking degradation
        n_ev = rng.integers(2, 4)
        ev = _rand_events(rng, n_ev, 3, 50, 4, 8)
        src_az = rng.uniform(-10, 10)
        src_el = rng.uniform(8, 16)
        gaze_az = src_az + rng.normal(0, 2.5, N)
        gaze_el = src_el + rng.normal(0, 2.5, N)
        g_true = np.clip(0.10 + 0.75 * _smooth_mask(t, ev), 0, 1)
        seg = rng.uniform(8, 15)             # slow decline segment
        t0 = rng.uniform(4, 12)
        decline = np.clip((t - t0) / seg, 0, 1) * (t < t0 + seg)
        c_true = np.clip(0.95 - 0.65 * decline, 0, 1)
        # dropouts
        d_ev = _rand_events(rng, rng.integers(1, 4), 10, 58, 0.2, 0.8)
        c_true[_interval_mask(t, d_ev)] = 0.0
        inhibit = _rand_events(rng, 1, 3, 55, 0.3, 0.6)
        eye_bias_mm = 0.0
    elif family == "S6":      # safety and faults
        n_ev = rng.integers(2, 4)
        ev = _rand_events(rng, n_ev, 3, 50, 4, 8)
        src_az = rng.uniform(-10, 10)
        src_el = rng.uniform(8, 16)
        gaze_az = src_az + rng.normal(0, 2.5, N)
        gaze_el = src_el + rng.normal(0, 2.5, N)
        g_true = np.clip(0.10 + 0.75 * _smooth_mask(t, ev), 0, 1)
        c_true = np.clip(0.90 + rng.normal(0, 0.03, N), 0, 1)
        inhibit = _rand_events(rng, rng.integers(2, 5), 3, 56, 0.25, 0.8)
        eye_bias_mm = float(rng.uniform(20, 30))
    else:
        raise ValueError(family)

    # ---------------- measured signals
    g_meas = np.clip(g_true + rng.normal(0, NOISE_SD["glare"], N), 0, 1)
    a_meas = alignment_from_dirs(gaze_az, gaze_el, src_az, src_el, rng,
                                 noise_sd=NOISE_SD["alignment"])
    c_meas = np.clip(c_true + rng.normal(0, NOISE_SD["confidence"], N), 0, 1)

    inhibit_mask = _interval_mask(t, inhibit)
    c_meas[inhibit_mask] = 0.0

    # invalid / stale frames (S5, S6): gaze frozen, confidence forced low
    stale_mask = np.zeros(N, dtype=bool)
    if family in ("S5", "S6"):
        s_ev = _rand_events(rng, rng.integers(1, 3), 10, 55, 0.5, 1.5)
        stale_mask = _interval_mask(t, s_ev)
        c_meas[stale_mask] = np.minimum(c_meas[stale_mask], 0.20)

    # eye-position bias (S6): measured gaze shifted laterally
    if eye_bias_mm > 0:
        bias_deg = np.degrees(np.arctan(eye_bias_mm / 1000.0 / D_W))
        biased = _interval_mask(t, _rand_events(rng, 2, 3, 55, 4, 8))
        gaze_az_m = gaze_az + bias_deg * biased
    else:
        gaze_az_m = gaze_az.copy()

    a_meas2 = alignment_from_dirs(gaze_az_m, gaze_el, src_az, src_el, rng,
                                  noise_sd=NOISE_SD["alignment"])

    # ---------------- true geometry ground truth
    th_true = np.empty(N)
    src_v = dir_vec(np.full(N, src_az), np.full(N, src_el))
    for sign in (-1.0, 1.0):
        d = dir_vec(gaze_az + sign * 0.8, gaze_el - 0.4)
        th = angular_sep(d, src_v)
        th_true = th if sign < 0 else np.minimum(th_true, th)
    a_true = np.exp(-th_true ** 2 / (2 * THETA_A ** 2))
    warranted = (g_true >= WARRANT_GLARE) & (a_true >= WARRANT_ALIGN)

    radius = np.clip(0.05 + 0.02 * (1 - c_meas), 0.05, 0.07)
    area = union_area(radius)

    return {"family": family, "seed": seed, "time_s": t,
            "g_true": g_true, "g_meas": g_meas, "a_true": a_true,
            "a_meas": a_meas2, "c_meas": c_meas, "theta_true_deg": np.degrees(th_true),
            "warranted": warranted.astype(int),
            "inhibit": inhibit_mask.astype(int), "stale": stale_mask.astype(int),
            "radius_m": radius, "dual_eye_union_area_m2": area}


# ------------------------------------------------------------- controllers

def controller_raw(kind, g, a, c):
    """Raw controller output (pre gate / filter / hysteresis)."""
    if kind == "fixed":
        return (g >= WARRANT_GLARE).astype(float)
    if kind == "crisp":
        return ((g >= WARRANT_GLARE) & (a >= 0.50)).astype(float)
    if kind == "fuzzy":
        return fuzzy_command(g, a, c)
    raise ValueError(kind)


def temporal_pipeline(raw, c, inhibit, gamma=GAMMA, tau=TAU,
                      theta_on=HYS_ON, theta_off=HYS_OFF, dt=DT,
                      gate=True, lowpass=True, hysteresis=True):
    """Confidence gate -> first-order low-pass -> hysteresis, sequentially.

    inhibit frames force alpha = 0 and reset the filter state and hysteresis.
    """
    n = len(raw)
    beta = 1.0 - math.exp(-dt / tau) if lowpass else 1.0
    u = np.where((c >= gamma) | (not gate), raw, 0.0)
    filt = np.zeros(n)
    state = np.zeros(n, dtype=int)
    alpha = np.zeros(n)
    for i in range(n):
        if inhibit[i]:
            continue                      # filt/state/alpha remain 0
        prev = filt[i - 1] if i else 0.0
        filt[i] = prev + beta * (u[i] - prev)
        if not hysteresis:
            state[i] = filt[i] > 0.0
        else:
            ps = state[i - 1] if i else 0
            state[i] = ps
            if ps == 0 and filt[i] >= theta_on:
                state[i] = 1
            elif ps == 1 and filt[i] <= theta_off:
                state[i] = 0
        alpha[i] = state[i] * filt[i]
    return alpha, filt, state


# ------------------------------------------------------------------ metrics

def _event_delays(t, warranted, alpha):
    """(activation delays, release delays) over contiguous warranted events."""
    act, rel = [], []
    d = np.diff(np.concatenate([[0], warranted.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        hit = np.where(alpha[s:] >= 0.5)[0]
        act.append(t[s + hit[0]] - t[s] if hit.size else np.nan)
        tail = alpha[e:]
        off = np.where(tail <= 0.1)[0]
        rel.append(t[e + off[0]] - t[e] if off.size else np.nan)
    return act, rel


def compute_metrics(sc, alpha, controller, stage):
    t = sc["time_s"]
    w = sc["warranted"].astype(bool)
    nw = ~w
    act = alpha > 0.05
    switches = int(np.count_nonzero(act[1:] != act[:-1]))
    tv = float(np.sum(np.abs(np.diff(alpha))))
    radius = sc["radius_m"]
    act_delay, rel_delay = _event_delays(t, sc["warranted"], alpha)
    return {
        "family": sc["family"], "seed": sc["seed"], "controller": controller,
        "stage": stage,
        "warranted_attenuation": float(np.mean(alpha[w])) if w.any() else np.nan,
        "warranted_error": float(np.mean(np.abs(1.0 - alpha[w]))) if w.any() else np.nan,
        "nuisance_attenuation": float(np.mean(alpha[nw])),
        "switch_count": switches,
        "total_variation": tv,
        "mean_active_area_m2": float(np.mean(radius[act] * 0 + sc["dual_eye_union_area_m2"][act])) if act.any() else 0.0,
        "time_average_area_m2": float(np.mean(act * sc["dual_eye_union_area_m2"])),
        "power_w": GLASS_POWER_DENSITY * float(np.mean(act * sc["dual_eye_union_area_m2"])),
        "activation_delay_s": float(np.nanmean(act_delay)) if act_delay else np.nan,
        "release_delay_s": float(np.nanmean(rel_delay)) if rel_delay else np.nan,
        "activation_detect_rate": float(np.nanmean([not np.isnan(x) for x in act_delay])) if act_delay else np.nan,
    }
