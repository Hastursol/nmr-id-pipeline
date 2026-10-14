"""Library search: rank the library compounds for one measured spectrum.

This is the counterpart of the ``__main__`` block of pSCNN.py, which pairs a
query spectrum with every standard and lists the predicted probabilities.  Here
any trained model (or the cosine baseline) can rank the K library compounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import cosine_scores, scores_cnn, scores_rf, scores_siamese
from .features import extract_features


def rank_library(spectrum: np.ndarray, runner, model_key: str = "siamese", seed: int = 0,
                 top: int = 10) -> pd.DataFrame:
    ds = runner.ds
    X = spectrum[None, :].astype(np.float32)
    if model_key == "cosine":
        s = cosine_scores(X, ds.refs)[0]
    elif model_key == "rf":
        rf, _ = runner.rf(seed)
        s = scores_rf(rf, extract_features(X, ds.grid), ds.n_classes)[0]
    elif model_key == "cnn":
        cnn, _ = runner.cnn(seed)
        s = scores_cnn(cnn, X, runner.device)[0]
    elif model_key == "siamese":
        sia, _ = runner.siamese(seed)
        s = scores_siamese(sia, X, ds.refs, runner.device, runner.cfg["siamese"]["loss"])[0]
    else:
        raise ValueError(model_key)
    order = np.argsort(-s)[:top]
    return pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "compound": [ds.names[i] for i in order],
        "smiles": [ds.smiles[i] for i in order],
        "key": [ds.keys[i] for i in order],
        "score": s[order],
    })
