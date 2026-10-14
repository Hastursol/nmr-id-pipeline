"""Configuration handling.

Every setting has a default here; a YAML file only needs the keys it changes.
The resolved configuration is written next to the results of every run so
that each table and figure in the thesis can be traced back to its settings.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "run_name": "default",
    "seed": 0,
    "device": "auto",  # auto | cpu | cuda | mps
    "paths": {
        "sdf": "data/raw/nmrshiftdb2withsignals.sd",
        "processed": "data/processed",
        "results": "results",
    },
    # ---------------------------------------------------------------- data
    "data": {
        "source": "nmrshiftdb2",  # nmrshiftdb2 | synthetic
        "n_compounds": 1000,  # size K of the compound library (classes)
        "min_peaks": 2,  # discard spectra with fewer resonances in range
        "max_records": None,  # read only the first N SD records (debugging)
        "prefer_multi_spectra": True,  # put compounds with >1 real 1H spectrum in the library first
        "replicates": 20,  # simulated measurements per compound
        "split": [0.70, 0.15, 0.15],  # train / val / test
        "include_reference_in_train": True,  # every model may train on the clean reference
        "synthetic_families": 0.5,  # synthetic mode: share of compounds derived from a parent
        "synthetic_real_frac": 0.2,  # synthetic mode: share with a second "measured" spectrum
    },
    # --------------------------------------------------------- spectrum grid
    "grid": {
        "ppm_min": 0.0,
        "ppm_max": 12.0,
        "n_bins": 2000,
        "hwhm_ppm": 0.012,  # Lorentzian half-width at half-maximum of the reference
        "larmor_mhz": 400.0,  # converts J couplings (Hz) to ppm for the reference
    },
    # ----------------------------------------- measurement simulator (replicates)
    "simulation": {
        "global_offset_sd": 0.005,  # referencing error, ppm
        "peak_jitter_sd": 0.010,  # per-resonance solvent/concentration drift, ppm
        "hwhm_scale": [0.7, 1.5],  # line-width multiplier range
        "intensity_jitter_sd": 0.10,  # log-normal jitter on each integral
        "j_jitter_sd": 0.02,  # relative jitter on coupling constants
        "drop_prob": 0.05,  # chance a resonance is missing (exchange, overlap)
        "impurity_prob": 0.30,  # chance of 1-2 residual solvent / water peaks
        "impurity_height": [0.05, 0.40],
        "noise_sd": [0.0, 0.01],  # Gaussian noise after max-normalisation
        "larmor_choices": [300.0, 400.0, 500.0, 600.0],
    },
    # ------------------------------------------------ on-the-fly augmentation
    "augment": {
        "p": 0.8,
        "noise_sd_max": 0.02,
        "max_shift_bins": 3,
        "scale": [0.8, 1.2],
    },
    # -------------------------------------------------------------- models
    "rf": {
        "backend": "compact",  # compact | sklearn (sklearn needs ~6 GB RAM for 1,000 classes)
        "n_estimators": 500,
        "max_depth": 30,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "n_jobs": -1,
    },
    "encoder": {
        "channels": [32, 64, 128, 256],
        "kernels": [7, 5, 3, 3],
        "pool_bins": 16,  # adaptive average pooling output length (1 = global average pooling)
        "bottleneck": 32,  # 1x1 conv channels before pooling (null = none)
        "embed_dim": 256,
        "dropout": 0.3,
    },
    "cnn": {
        "dropout": 0.5,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "batch_size": 64,
        "epochs": 50,
        "patience": 8,
        "min_batches_per_epoch": 50,  # small training sets are cycled (learning curves)
    },
    "siamese": {
        "shared_weights": True,  # False reproduces the pseudo-Siamese idea of Wei et al. (2022)
        "loss": "bce",  # bce | contrastive
        "margin": 1.0,  # contrastive margin (embeddings are L2-normalised)
        "ref_partner_prob": 0.5,  # partner is the clean reference (else another replicate)
        "hard_negative_frac": 0.5,
        "hard_pool": 20,  # hard negatives come from the N most similar references
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "batch_size": 64,
        "epochs": 40,
        "patience": 8,
        "min_batches_per_epoch": 50,
        "max_val_queries": 2000,
    },
    # ---------------------------------------------------------- experiments
    "experiments": {
        "seeds": [0],  # use [0, 1, 2] for the final RQ1 table (mean +/- std)
        "top_k": [1, 5],
        "noise_levels": [0.0, 0.01, 0.02, 0.05, 0.10],
        "shift_ppm": [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
        "train_fractions": [0.10, 0.25, 0.50, 0.75, 1.00],
        "models": ["cosine", "rf", "cnn", "siamese"],
    },
}


def deep_update(base: dict, update: dict) -> dict:
    """Recursively merge ``update`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = deep_update(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def parse_override(text: str) -> dict:
    """Turn ``cnn.epochs=5`` into ``{"cnn": {"epochs": 5}}`` (value parsed as YAML)."""
    key, _, raw = text.partition("=")
    value = yaml.safe_load(raw)
    out: dict = {}
    node = out
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return out


def run_dir(cfg: dict) -> Path:
    path = Path(cfg["paths"]["results"]) / cfg["run_name"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(cfg: dict) -> Path:
    path = Path(cfg["paths"]["processed"])
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{cfg['run_name']}_dataset.npz"


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
