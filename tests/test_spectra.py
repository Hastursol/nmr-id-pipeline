import numpy as np

from nmrid.augment import SpectrumAugmenter, shift_ppm
from nmrid.spectra import Grid, Resonance, bin_continuous, exclude_regions, expand_multiplet, render

GRID = Grid()


def test_render_conserves_integral():
    spec = render([Resonance(3.0, 3.0), Resonance(7.2, 5.0)], GRID, hwhm=0.01, normalize=False)
    assert abs(spec.sum() - 8.0) < 0.05


def test_render_normalised_and_positioned():
    spec = render([Resonance(2.0, 1.0)], GRID)
    assert np.isclose(spec.max(), 1.0)
    assert abs(GRID.centers[spec.argmax()] - 2.0) < GRID.width


def test_triplet_follows_n_plus_one_rule():
    pos, w = expand_multiplet(Resonance(1.0, 3.0, ((7.0, 2),)), larmor_mhz=400.0)
    assert np.allclose(w, [0.75, 1.5, 0.75])
    assert np.allclose(np.diff(pos), 7.0 / 400.0)


def test_bin_continuous_matches_peak_position():
    ppm = np.linspace(12.5, -0.5, 65536)  # descending, like a Bruker spectrum
    y = 1.0 / (1 + ((ppm - 4.5) / 0.003) ** 2)
    spec = bin_continuous(ppm, y, GRID)
    assert abs(GRID.centers[spec.argmax()] - 4.5) < GRID.width


def test_exclude_regions_blanks_window():
    spec = render([Resonance(7.26, 1.0), Resonance(2.0, 1.0)], GRID)
    out = exclude_regions(spec, GRID, [[7.2, 7.3]])
    assert out[(GRID.centers > 7.2) & (GRID.centers < 7.3)].max() == 0
    assert np.isclose(out.max(), 1.0)


def test_augmenter_keeps_shape_and_zero_fills():
    import torch

    x = torch.zeros(4, 100)
    x[:, 0] = 1.0
    aug = SpectrumAugmenter(p=1.0, noise_sd_max=0.0, max_shift_bins=3, scale=(1.0, 1.0))
    out = aug(x)
    assert out.shape == x.shape
    assert out[:, -3:].abs().max() == 0  # nothing wrapped around to the other end


def test_shift_ppm_moves_peak():
    spec = render([Resonance(5.0, 1.0)], GRID)[None, :]
    moved = shift_ppm(spec, 0.03, GRID.width)
    assert abs(GRID.centers[moved.argmax()] - 5.03) < GRID.width
