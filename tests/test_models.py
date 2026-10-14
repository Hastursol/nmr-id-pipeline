import numpy as np
import torch

from nmrid.features import N_FEATURES, extract_features
from nmrid.models import SiameseNet, SpectrumCNN, SpectrumEncoder, count_parameters
from nmrid.spectra import Grid, Resonance, render


def test_feature_vector_layout():
    grid = Grid()
    X = np.stack([render([Resonance(1.0, 3), Resonance(7.3, 5)], grid)])
    F = extract_features(X, grid)
    assert F.shape == (1, N_FEATURES) == (1, 145)
    peaks = F[0, 100:120]
    found = sorted(p for p in peaks if p >= 0)
    assert len(found) == 2 and abs(found[0] - 1.0) < 0.01 and abs(found[1] - 7.3) < 0.01
    assert F[0, 140] == 2  # peak count


def test_encoder_and_cnn_shapes():
    x = torch.rand(3, 2000)
    enc = SpectrumEncoder()
    assert enc(x).shape == (3, 256)
    cnn = SpectrumCNN(n_classes=50)
    assert cnn(x).shape == (3, 50)
    gap = SpectrumEncoder(pool_bins=1, bottleneck=None)
    assert gap(x).shape == (3, 256)


def test_siamese_shared_vs_pseudo():
    shared, pseudo = SiameseNet(shared_weights=True), SiameseNet(shared_weights=False)
    enc = count_parameters(SpectrumEncoder())
    assert count_parameters(pseudo) - count_parameters(shared) == enc
    logit, eq, er = shared(torch.rand(2, 2000), torch.rand(2, 2000))
    assert logit.shape == (2,) and torch.allclose(eq.norm(dim=1), torch.ones(2), atol=1e-5)


def test_backward_pass():
    cnn = SpectrumCNN(n_classes=5)
    loss = torch.nn.functional.cross_entropy(cnn(torch.rand(4, 2000)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert all(p.grad is not None for p in cnn.parameters())
