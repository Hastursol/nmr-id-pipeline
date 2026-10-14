"""Measurement simulator and synthetic compound library.

Why a simulator?  nmrshiftdb2 stores (mostly) one 1H peak list per compound.
A closed-set classifier needs several example spectra of every class, so each
reference peak list is "re-measured" many times with the kinds of variation
that separate two real recordings of the same compound:

* a global referencing offset and per-resonance drift (solvent, concentration,
  temperature), comparable to the shift variation Wei et al. (2022, Fig. 6a)
  measured between mixtures and pure standards;
* a different spectrometer frequency, which changes multiplet widths in ppm;
* line-width and integral variation;
* occasionally missing resonances (exchangeable protons, overlap);
* residual solvent and water peaks (values from Fulmer et al., 2010);
* Gaussian noise.

The synthetic library exists only so the whole pipeline can be exercised in a
few minutes without downloading anything.  Numbers produced on synthetic data
are smoke-test numbers, not thesis results.
"""
from __future__ import annotations

import numpy as np

from .spectra import Grid, Resonance, normalize_max, render

# Residual solvent / water 1H shifts (ppm), Fulmer et al., Organometallics 2010, 29, 2176.
SOLVENT_PEAKS = {
    "CHCl3 in CDCl3": 7.26,
    "H2O in CDCl3": 1.56,
    "DMSO-d5 in DMSO-d6": 2.50,
    "H2O in DMSO-d6": 3.33,
    "acetone-d5": 2.05,
    "H2O in acetone-d6": 2.84,
    "CHD2OD in CD3OD": 3.31,
    "H2O in CD3OD": 4.87,
    "HDO in D2O": 4.79,
    "C6D5H in C6D6": 7.16,
    "TMS": 0.00,
}


def simulate_measurement(resonances, grid: Grid, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Return one simulated, max-normalised measurement of a compound."""
    s = cfg["simulation"]
    base_hwhm = float(cfg["grid"]["hwhm_ppm"])
    larmor = float(rng.choice(s["larmor_choices"]))
    offset = rng.normal(0.0, s["global_offset_sd"])

    measured = []
    for r in resonances:
        if len(resonances) >= 4 and rng.random() < s["drop_prob"]:
            continue
        couplings = tuple(
            (j * (1.0 + rng.normal(0.0, s["j_jitter_sd"])), n) for j, n in r.couplings
        )
        measured.append(
            Resonance(
                shift=r.shift + offset + rng.normal(0.0, s["peak_jitter_sd"]),
                n_h=r.n_h * float(np.exp(rng.normal(0.0, s["intensity_jitter_sd"]))),
                couplings=couplings,
                broad=r.broad,
            )
        )

    hwhm = base_hwhm * rng.uniform(*s["hwhm_scale"])
    spec = render(measured, grid, hwhm=hwhm, larmor_mhz=larmor, normalize=True)

    if rng.random() < s["impurity_prob"]:
        names = list(SOLVENT_PEAKS)
        for name in rng.choice(names, size=rng.integers(1, 3), replace=False):
            imp = render([Resonance(SOLVENT_PEAKS[name], 1.0)], grid, hwhm=hwhm, normalize=True)
            spec = spec + rng.uniform(*s["impurity_height"]) * imp
        spec = normalize_max(spec)

    noise_sd = rng.uniform(*s["noise_sd"])
    if noise_sd > 0:
        spec = spec + rng.normal(0.0, noise_sd, size=spec.shape)
    return spec.astype(np.float32)


# ----------------------------------------------------------------- synthetic
# (low ppm, high ppm, probability, possible proton counts, typical partners)
_REGIONS = [
    (0.80, 1.50, 0.28, [3, 2, 6, 9], [0, 1, 2, 3]),  # alkyl CH3 / CH2
    (1.50, 2.60, 0.20, [3, 2], [0, 1, 2]),  # allylic, alpha to C=O
    (3.20, 4.40, 0.18, [3, 2, 1], [0, 1, 2]),  # next to O / N
    (4.50, 6.50, 0.07, [1, 2], [1]),  # vinylic
    (6.50, 8.30, 0.21, [1, 2], [1, 2]),  # aromatic
    (9.40, 10.10, 0.03, [1], [0]),  # aldehyde
    (10.50, 12.00, 0.03, [1], [0]),  # carboxylic acid (broad)
]


def _random_resonance(rng: np.random.Generator) -> Resonance:
    probs = np.array([r[2] for r in _REGIONS])
    lo, hi, _, counts, partners = _REGIONS[rng.choice(len(_REGIONS), p=probs / probs.sum())]
    shift = rng.uniform(lo, hi)
    n_h = float(rng.choice(counts))
    n_partners = int(rng.choice(partners))
    couplings: tuple = ()
    if n_partners > 0:
        j = rng.uniform(7.5, 8.5) if lo >= 6.5 else rng.uniform(6.0, 7.5)
        couplings = ((j, n_partners),)
        if rng.random() < 0.15:  # a second, smaller coupling (dd, dt, ...)
            couplings = couplings + ((rng.uniform(1.5, 3.0), 1),)
    return Resonance(shift, n_h, couplings, broad=lo >= 10.5)


def random_compound(rng: np.random.Generator) -> list:
    n = int(rng.integers(2, 9))
    return sorted((_random_resonance(rng) for _ in range(n)), key=lambda r: r.shift)


def _derive(parent: list, rng: np.random.Generator) -> list:
    """A structural analogue: same skeleton, shifted environments, maybe +/- one group."""
    child = [
        Resonance(r.shift + rng.normal(0.0, 0.08), r.n_h, r.couplings, r.broad) for r in parent
    ]
    if len(child) > 2 and rng.random() < 0.4:
        child.pop(int(rng.integers(len(child))))
    if rng.random() < 0.4:
        child.append(_random_resonance(rng))
    return sorted(child, key=lambda r: r.shift)


def synthetic_library(n: int, cfg: dict, rng: np.random.Generator) -> list[dict]:
    """Return compound records in the same format as the nmrshiftdb2 loader."""
    family_frac = float(cfg["data"]["synthetic_families"])
    real_frac = float(cfg["data"]["synthetic_real_frac"])
    compounds: list[dict] = []
    parents: list[list] = []
    while len(compounds) < n:
        if parents and rng.random() < family_frac:
            res = _derive(parents[int(rng.integers(len(parents)))], rng)
        else:
            res = random_compound(rng)
            parents.append(res)
        extra = []
        if rng.random() < real_frac:  # a second "real" measurement in another solvent
            extra.append(
                {
                    "resonances": [
                        Resonance(r.shift + rng.normal(0.0, 0.02), r.n_h, r.couplings, r.broad)
                        for r in res
                    ],
                    "larmor": float(rng.choice([300.0, 400.0, 500.0, 600.0])),
                    "meta": {"note": "synthetic second measurement"},
                }
            )
        idx = len(compounds)
        compounds.append(
            {
                "key": f"SYN{idx:05d}",
                "name": f"synthetic-{idx:05d}",
                "smiles": "",
                "spectra": [{"resonances": res, "larmor": 400.0, "meta": {}}] + extra,
            }
        )
    return compounds
