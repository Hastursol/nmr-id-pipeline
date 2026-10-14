"""On-the-fly augmentation of binned spectra (applied to each training batch).

1. Gaussian noise injection       simulates detector noise
2. random chemical-shift shift    moves the whole spectrum by a few bins (zero fill,
                                  no wrap-around from 12 ppm back to 0 ppm)
3. intensity scaling              multiplies the spectrum by a factor near one, so the
                                  model does not rely on the tallest peak being exactly 1
"""
from __future__ import annotations

import torch


class SpectrumAugmenter:
    def __init__(self, p: float = 0.8, noise_sd_max: float = 0.02, max_shift_bins: int = 3,
                 scale=(0.8, 1.2)):
        self.p = float(p)
        self.noise_sd_max = float(noise_sd_max)
        self.max_shift = int(max_shift_bins)
        self.scale = tuple(float(s) for s in scale)

    @classmethod
    def from_config(cls, cfg: dict) -> "SpectrumAugmenter":
        a = cfg["augment"]
        return cls(a["p"], a["noise_sd_max"], a["max_shift_bins"], a["scale"])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return x
        b, n = x.shape
        active = (torch.rand(b, 1, device=x.device) < self.p).float()
        out = x
        if self.max_shift > 0:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (b, 1), device=x.device)
            shift = shift * active.long()
            idx = torch.arange(n, device=x.device)[None, :] - shift
            valid = (idx >= 0) & (idx < n)
            out = torch.gather(out, 1, idx.clamp(0, n - 1)) * valid
        lo, hi = self.scale
        factor = lo + (hi - lo) * torch.rand(b, 1, device=x.device)
        out = out * (1 + active * (factor - 1))
        sd = self.noise_sd_max * torch.rand(b, 1, device=x.device) * active
        return out + sd * torch.randn_like(out)


def add_noise(X, sd: float, seed: int = 0):
    """Deterministic Gaussian noise for the robustness experiment (numpy in, numpy out)."""
    import numpy as np

    if sd <= 0:
        return X
    rng = np.random.default_rng(seed)
    return (X + rng.normal(0.0, sd, size=X.shape)).astype(np.float32)


def shift_ppm(X, delta_ppm: float, bin_width: float):
    """Shift whole spectra by ``delta_ppm`` using linear interpolation between bins."""
    import numpy as np

    if delta_ppm == 0:
        return X
    n = X.shape[1]
    src = np.arange(n) - delta_ppm / bin_width
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = np.interp(src, np.arange(n), X[i], left=0.0, right=0.0)
    return out
