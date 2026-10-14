"""Hand-crafted spectral descriptors for the Random Forest baseline (145 features).

    group             size  content
    histogram          100  summed intensity in 100 equal windows of 0.12 ppm
    peak_positions      20  ppm of the 20 tallest peaks, listed in ascending ppm (-1 = no peak)
    peak_heights        20  heights of those peaks, in the same order (0 = no peak)
    peak_count           1  number of detected peaks
    moments              4  mean, standard deviation, skewness and excess kurtosis of the
                            spectrum treated as a distribution over the ppm axis
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .spectra import Grid

GROUPS = [("histogram", 100), ("peak_positions", 20), ("peak_heights", 20),
          ("peak_count", 1), ("moments", 4)]
N_FEATURES = sum(n for _, n in GROUPS)


def feature_names() -> list[str]:
    names = [f"hist_{i:03d}" for i in range(100)]
    names += [f"peak_pos_{i:02d}" for i in range(20)]
    names += [f"peak_height_{i:02d}" for i in range(20)]
    names += ["peak_count", "ppm_mean", "ppm_std", "ppm_skew", "ppm_kurtosis"]
    return names


def feature_groups() -> np.ndarray:
    """Group label of every feature column (for group-level importances)."""
    return np.concatenate([[g] * n for g, n in GROUPS])


def extract_features(X: np.ndarray, grid: Grid, n_hist: int = 100, n_peaks: int = 20,
                     min_height: float = 0.02, min_prominence: float = 0.02) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, np.float32))
    N, L = X.shape
    pos = np.clip(X, 0.0, None)
    centers = grid.centers

    edges = np.linspace(0, L, n_hist + 1).astype(int)
    hist = np.add.reduceat(pos, edges[:-1], axis=1)

    peak_pos = np.full((N, n_peaks), -1.0, np.float32)
    peak_h = np.zeros((N, n_peaks), np.float32)
    count = np.zeros((N, 1), np.float32)
    for i in range(N):
        idx, props = find_peaks(X[i], height=min_height, prominence=min_prominence)
        count[i, 0] = len(idx)
        if len(idx):
            top = idx[np.argsort(props["peak_heights"])[::-1][:n_peaks]]
            top = np.sort(top)
            peak_pos[i, : len(top)] = centers[top]
            peak_h[i, : len(top)] = X[i, top]

    w = pos / np.maximum(pos.sum(axis=1, keepdims=True), 1e-12)
    mean = (w * centers).sum(axis=1)
    dev = centers[None, :] - mean[:, None]
    var = (w * dev**2).sum(axis=1)
    std = np.sqrt(np.maximum(var, 1e-12))
    skew = (w * dev**3).sum(axis=1) / std**3
    kurt = (w * dev**4).sum(axis=1) / std**4 - 3.0
    moments = np.stack([mean, std, skew, kurt], axis=1)

    return np.concatenate([hist, peak_pos, peak_h, count, moments], axis=1).astype(np.float32)
