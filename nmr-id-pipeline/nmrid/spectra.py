"""Spectrum representation: the shared ppm grid, resonances, Lorentzian rendering.

Every spectrum in the pipeline, whether reconstructed from an nmrshiftdb2 peak
list, simulated, or read from a Bruker experiment, ends up as a vector of
``n_bins`` intensities on the same ppm grid, normalised to a maximum of one.

Index 0 of the vector is the low-ppm end (0 ppm).  Plots invert the axis so
that figures follow the usual NMR convention (high ppm on the left).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np


@dataclass(frozen=True)
class Grid:
    ppm_min: float = 0.0
    ppm_max: float = 12.0
    n_bins: int = 2000

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(self.ppm_min, self.ppm_max, self.n_bins + 1)

    @property
    def centers(self) -> np.ndarray:
        e = self.edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def width(self) -> float:
        return (self.ppm_max - self.ppm_min) / self.n_bins

    @classmethod
    def from_config(cls, cfg: dict) -> "Grid":
        g = cfg["grid"]
        return cls(float(g["ppm_min"]), float(g["ppm_max"]), int(g["n_bins"]))


@dataclass
class Resonance:
    """One chemically distinct proton environment.

    shift      chemical shift in ppm
    n_h        number of protons (the integral)
    couplings  first-order couplings as ((J in Hz, number of equivalent partners), ...)
    broad      exchange-broadened signal (OH, NH, COOH); rendered three times wider
    """

    shift: float
    n_h: float = 1.0
    couplings: tuple = field(default_factory=tuple)
    broad: bool = False


def expand_multiplet(res: Resonance, larmor_mhz: float) -> tuple[np.ndarray, np.ndarray]:
    """Split a resonance into its first-order lines (n + 1 rule, binomial intensities).

    Returns line positions (ppm) and line weights that sum to ``res.n_h``.
    """
    positions = np.array([res.shift], dtype=float)
    weights = np.array([1.0])
    for j_hz, n_partners in res.couplings:
        n_partners = int(n_partners)
        if n_partners <= 0 or j_hz <= 0:
            continue
        j_ppm = float(j_hz) / float(larmor_mhz)
        offsets = (np.arange(n_partners + 1) - n_partners / 2.0) * j_ppm
        pattern = np.array([comb(n_partners, k) for k in range(n_partners + 1)], float)
        pattern /= pattern.sum()
        positions = (positions[:, None] + offsets[None, :]).ravel()
        weights = (weights[:, None] * pattern[None, :]).ravel()
    return positions, weights * float(res.n_h)


def lines_from_resonances(resonances, larmor_mhz: float, hwhm: float):
    pos, wts, hws = [], [], []
    for r in resonances:
        p, w = expand_multiplet(r, larmor_mhz)
        pos.append(p)
        wts.append(w)
        hws.append(np.full(p.shape, hwhm * (3.0 if r.broad else 1.0)))
    if not pos:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(pos), np.concatenate(wts), np.concatenate(hws)


def render(resonances, grid: Grid, hwhm: float = 0.012, larmor_mhz: float = 400.0,
           normalize: bool = True) -> np.ndarray:
    """Render resonances as a binned spectrum.

    Each line is a Lorentzian with half-width ``hwhm`` (ppm).  Instead of
    sampling the Lorentzian at bin centres, the exact area falling inside each
    bin is computed from its cumulative distribution,
    ``F(x) = 1/2 + arctan((x - x0) / gamma) / pi``, so the vector conserves the
    integral of every peak however narrow it is relative to the bin width.
    """
    pos, wts, hws = lines_from_resonances(resonances, larmor_mhz, hwhm)
    spec = np.zeros(grid.n_bins, dtype=np.float64)
    if pos.size == 0:
        return spec.astype(np.float32)
    edges = grid.edges
    cdf = np.arctan((edges[None, :] - pos[:, None]) / hws[:, None]) / np.pi
    spec = (wts[:, None] * np.diff(cdf, axis=1)).sum(axis=0)
    if normalize:
        spec = normalize_max(spec)
    return spec.astype(np.float32)


def normalize_max(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    return x / m if m > 0 else x


def bin_continuous(ppm: np.ndarray, intensity: np.ndarray, grid: Grid,
                   clip_negative: bool = True, normalize: bool = True) -> np.ndarray:
    """Integrate a densely sampled experimental spectrum into the grid bins.

    ``ppm`` may be ascending or descending.  Points outside the grid are ignored.
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(intensity, dtype=float).real
    if clip_negative:
        y = np.clip(y, 0.0, None)
    step = np.abs(np.median(np.diff(ppm))) if ppm.size > 1 else 1.0
    spec, _ = np.histogram(ppm, bins=grid.edges, weights=y * step)
    if normalize:
        spec = normalize_max(spec)
    return spec.astype(np.float32)


def exclude_regions(spec: np.ndarray, grid: Grid, regions, renormalize: bool = True) -> np.ndarray:
    """Zero out ppm windows, e.g. [[7.20, 7.30]] for residual CHCl3."""
    out = spec.copy()
    c = grid.centers
    for lo, hi in regions or []:
        out[(c >= min(lo, hi)) & (c <= max(lo, hi))] = 0.0
    return normalize_max(out) if renormalize else out


def resonances_in_range(resonances, grid: Grid) -> list:
    return [r for r in resonances if grid.ppm_min <= r.shift <= grid.ppm_max]
