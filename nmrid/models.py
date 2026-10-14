"""Neural network models (PyTorch).

SpectrumEncoder   the convolutional feature extractor shared by both networks
SpectrumCNN       encoder + classification head (Model 2, closed-set classifier)
SiameseNet        two encoders + similarity head (Model 3, metric learning)

Design notes relative to pSCNN (Wei et al., 2022):

* pSCNN stacks 6 Conv1D(32 filters, kernel 5) + max-pool layers per branch and
  flattens the result.  Here four blocks widen from 32 to 256 filters with
  batch normalisation, as specified in the thesis plan.
* Position matters in NMR: a methyl singlet at 1.2 ppm and one at 3.7 ppm have
  the same local shape but mean different chemistry.  Global average pooling
  (pool_bins = 1) discards where a pattern occurred, so the encoder pools to
  ``pool_bins`` segments (16 segments of 0.75 ppm by default) and keeps their
  order.  Setting pool_bins = 1 reproduces plain global average pooling for the
  ablation.
* pSCNN's two branches have separate weights ("pseudo-Siamese") because one
  branch sees a mixture and the other a pure compound.  Here both inputs are
  single-compound spectra, so the weights are shared by default;
  ``shared_weights=False`` gives the pseudo-Siamese variant for comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, kernel: int, pool: bool):
        layers = [nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2, bias=False),
                  nn.BatchNorm1d(c_out), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool1d(2))
        super().__init__(*layers)


class SpectrumEncoder(nn.Module):
    """(batch, n_bins) spectrum -> (batch, embed_dim) feature vector."""

    def __init__(self, channels=(32, 64, 128, 256), kernels=(7, 5, 3, 3), pool_bins: int = 16,
                 bottleneck: int | None = 32, embed_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        blocks, c_in = [], 1
        for i, (c_out, k) in enumerate(zip(channels, kernels)):
            blocks.append(ConvBlock(c_in, c_out, k, pool=i < len(channels) - 1))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        if bottleneck:
            self.reduce = nn.Sequential(nn.Conv1d(c_in, bottleneck, 1, bias=False),
                                        nn.BatchNorm1d(bottleneck), nn.ReLU(inplace=True))
            c_in = bottleneck
        else:
            self.reduce = nn.Identity()
        self.pool = nn.AdaptiveAvgPool1d(pool_bins)
        self.project = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                     nn.Linear(c_in * pool_bins, embed_dim), nn.ReLU(inplace=True))
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.project(self.pool(self.reduce(self.features(x))))


class SpectrumCNN(nn.Module):
    def __init__(self, n_classes: int, encoder_kwargs: dict | None = None, dropout: float = 0.5):
        super().__init__()
        self.encoder = SpectrumEncoder(**(encoder_kwargs or {}))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.embed_dim, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SiameseNet(nn.Module):
    """Scores whether a query spectrum and a reference spectrum come from one compound."""

    def __init__(self, encoder_kwargs: dict | None = None, shared_weights: bool = True):
        super().__init__()
        self.query_encoder = SpectrumEncoder(**(encoder_kwargs or {}))
        self.ref_encoder = (self.query_encoder if shared_weights
                            else SpectrumEncoder(**(encoder_kwargs or {})))
        d = self.query_encoder.embed_dim
        # symmetric pair features |a - b| and a * b, then a small MLP -> logit
        self.head = nn.Sequential(nn.Linear(2 * d, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.2), nn.Linear(128, 1))
        self.shared_weights = shared_weights

    def embed_query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_encoder(x), dim=-1)

    def embed_ref(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.ref_encoder(x), dim=-1)

    def pair_logit(self, eq: torch.Tensor, er: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([(eq - er).abs(), eq * er], dim=-1)).squeeze(-1)

    def forward(self, query: torch.Tensor, ref: torch.Tensor):
        eq, er = self.embed_query(query), self.embed_ref(ref)
        return self.pair_logit(eq, er), eq, er


def contrastive_loss(eq: torch.Tensor, er: torch.Tensor, same: torch.Tensor,
                     margin: float = 1.0) -> torch.Tensor:
    """Hadsell et al. (2006): y d^2 + (1 - y) max(0, m - d)^2."""
    d = (eq - er).norm(dim=-1)
    return (same * d.pow(2) + (1 - same) * F.relu(margin - d).pow(2)).mean()


def count_parameters(model: nn.Module) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen and p.requires_grad:
            seen.add(id(p))
            total += p.numel()
    return total


def encoder_kwargs_from_config(cfg: dict) -> dict:
    e = cfg["encoder"]
    return {"channels": tuple(e["channels"]), "kernels": tuple(e["kernels"]),
            "pool_bins": int(e["pool_bins"]), "bottleneck": e.get("bottleneck"),
            "embed_dim": int(e["embed_dim"]), "dropout": float(e["dropout"])}
