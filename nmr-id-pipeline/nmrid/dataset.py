"""Build the benchmark dataset used by every experiment.

Two evaluation protocols are prepared from one set of K library compounds:

Closed-set identification (RQ1-RQ4, all models)
    The question "which of the K known library compounds produced this
    spectrum?".  Each compound has one clean reference spectrum plus R
    simulated measurements (replicates).  The replicates of every compound are
    split 70/15/15 into train / validation / test, so every class appears in
    every split (a classifier cannot predict a class it has never seen) while no
    individual test spectrum is ever used for training.  Compounds with a second,
    genuinely measured nmrshiftdb2 spectrum give an extra "real re-measurement"
    test set that involves no simulation at all.

Open-set retrieval (optional experiment, Siamese vs. baselines)
    The compounds themselves are split 70/15/15.  A model trained on the
    training compounds has to find compounds it has never seen by comparing
    spectra against the reference library.  This is where the compound-level
    split belongs, and only similarity-based models can take part.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .simulate import simulate_measurement, synthetic_library
from .spectra import Grid, render

TRAIN, VAL, TEST = 0, 1, 2


@dataclass
class NMRDataset:
    X: np.ndarray  # (N, n_bins) simulated measurements
    y: np.ndarray  # (N,) compound index
    rep: np.ndarray  # (N,) replicate number within the compound
    split: np.ndarray  # (N,) closed-set split of the replicates (0/1/2)
    refs: np.ndarray  # (K, n_bins) clean reference spectra
    compound_split: np.ndarray  # (K,) open-set split of the compounds (0/1/2)
    names: list
    keys: list
    smiles: list
    real_X: np.ndarray  # (M, n_bins) genuine second measurements (may be empty)
    real_y: np.ndarray  # (M,)
    grid: Grid
    info: dict = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return self.refs.shape[0]

    # ---------------------------------------------------------------- views
    def closed_set(self, part: int, train_fraction: float = 1.0,
                   include_reference: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Spectra and labels of one closed-set split.

        ``train_fraction`` keeps, for every compound, only the first
        ceil(fraction x n_train_replicates) training replicates (learning curves).
        ``include_reference`` appends the clean reference of every compound.
        """
        mask = self.split == part
        if part == TRAIN and train_fraction < 1.0:
            keep = np.zeros_like(mask)
            for c in range(self.n_classes):
                idx = np.flatnonzero(mask & (self.y == c))
                n_keep = max(1, int(np.ceil(train_fraction * len(idx))))
                keep[idx[:n_keep]] = True
            mask = keep
        X, y = self.X[mask], self.y[mask]
        if include_reference:
            X = np.concatenate([X, self.refs])
            y = np.concatenate([y, np.arange(self.n_classes)])
        return X, y

    def compounds_in(self, part: int) -> np.ndarray:
        return np.flatnonzero(self.compound_split == part)

    def open_set(self, part: int) -> tuple[np.ndarray, np.ndarray]:
        """All replicates of the compounds in one open-set split."""
        mask = np.isin(self.y, self.compounds_in(part))
        return self.X[mask], self.y[mask]

    # ------------------------------------------------------------ save/load
    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            X=self.X, y=self.y, rep=self.rep, split=self.split, refs=self.refs,
            compound_split=self.compound_split,
            names=np.array(self.names, dtype=object), keys=np.array(self.keys, dtype=object),
            smiles=np.array(self.smiles, dtype=object),
            real_X=self.real_X, real_y=self.real_y,
            grid=np.array([self.grid.ppm_min, self.grid.ppm_max, self.grid.n_bins]),
            info=np.array([self.info], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NMRDataset":
        d = np.load(path, allow_pickle=True)
        g = d["grid"]
        return cls(
            X=d["X"], y=d["y"], rep=d["rep"], split=d["split"], refs=d["refs"],
            compound_split=d["compound_split"], names=list(d["names"]), keys=list(d["keys"]),
            smiles=list(d["smiles"]), real_X=d["real_X"], real_y=d["real_y"],
            grid=Grid(float(g[0]), float(g[1]), int(g[2])), info=d["info"][0],
        )


def _split_counts(n: int, fractions) -> tuple[int, int, int]:
    f = np.asarray(fractions, float) / np.sum(fractions)
    n_val = max(1, int(round(f[1] * n)))
    n_test = max(1, int(round(f[2] * n)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Too few items ({n}) for a train/val/test split")
    return n_train, n_val, n_test


def select_library(compounds: list[dict], n: int, prefer_multi: bool,
                   rng: np.random.Generator) -> list[dict]:
    order = rng.permutation(len(compounds))
    if prefer_multi:
        multi = [i for i in order if len(compounds[i]["spectra"]) > 1]
        single = [i for i in order if len(compounds[i]["spectra"]) == 1]
        order = np.array(multi + single, dtype=int)
    chosen = sorted(order[: min(n, len(order))].tolist())
    return [compounds[i] for i in chosen]


def build_dataset(compounds: list[dict], cfg: dict, seed: int | None = None,
                  progress: bool = True) -> NMRDataset:
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    grid = Grid.from_config(cfg)
    d = cfg["data"]
    lib = select_library(compounds, int(d["n_compounds"]), bool(d["prefer_multi_spectra"]), rng)
    K, R = len(lib), int(d["replicates"])
    n_train, n_val, n_test = _split_counts(R, d["split"])
    hwhm, larmor = float(cfg["grid"]["hwhm_ppm"]), float(cfg["grid"]["larmor_mhz"])

    refs = np.zeros((K, grid.n_bins), np.float32)
    X = np.zeros((K * R, grid.n_bins), np.float32)
    y = np.repeat(np.arange(K), R)
    rep = np.tile(np.arange(R), K)
    split = np.tile(np.array([TRAIN] * n_train + [VAL] * n_val + [TEST] * n_test), K)
    real_X, real_y = [], []

    iterator = range(K)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Simulating measurements", unit="compound")
        except Exception:  # pragma: no cover
            pass
    for c in iterator:
        reference = lib[c]["spectra"][0]
        refs[c] = render(reference["resonances"], grid, hwhm=hwhm,
                         larmor_mhz=reference.get("larmor") or larmor)
        for r in range(R):
            X[c * R + r] = simulate_measurement(reference["resonances"], grid, cfg, rng)
        for other in lib[c]["spectra"][1:]:
            real_X.append(render(other["resonances"], grid, hwhm=hwhm,
                                 larmor_mhz=other.get("larmor") or larmor))
            real_y.append(c)

    # Replicates are drawn i.i.d., so assigning the first n_train of each compound
    # to training (and taking the first few for learning curves) is an unbiased choice.
    n_open = _split_counts(K, d["split"])
    perm = rng.permutation(K)
    compound_split = np.zeros(K, np.int8)
    compound_split[perm[n_open[0]: n_open[0] + n_open[1]]] = VAL
    compound_split[perm[n_open[0] + n_open[1]:]] = TEST

    info = {
        "source": d["source"], "n_compounds": K, "replicates": R,
        "replicate_split": [n_train, n_val, n_test], "compound_split": list(n_open),
        "n_real_remeasurements": len(real_y), "seed": seed,
    }
    return NMRDataset(
        X=X, y=y.astype(np.int64), rep=rep.astype(np.int16), split=split.astype(np.int8),
        refs=refs, compound_split=compound_split,
        names=[c["name"] for c in lib], keys=[c["key"] for c in lib],
        smiles=[c.get("smiles", "") for c in lib],
        real_X=np.array(real_X, np.float32).reshape(-1, grid.n_bins),
        real_y=np.array(real_y, np.int64), grid=grid, info=info,
    )


def load_compounds(cfg: dict, progress: bool = True) -> tuple[list[dict], dict]:
    """Compound records from nmrshiftdb2 or from the synthetic generator."""
    grid = Grid.from_config(cfg)
    d = cfg["data"]
    if d["source"] == "synthetic":
        rng = np.random.default_rng(cfg["seed"] + 1000)
        comps = synthetic_library(int(d["n_compounds"]), cfg, rng)
        return comps, {"source": "synthetic", "compounds": len(comps)}
    from .io_nmrshiftdb import load_nmrshiftdb

    sdf = Path(cfg["paths"]["sdf"])
    if not sdf.exists():
        raise FileNotFoundError(
            f"{sdf} not found. Download nmrshiftdb2withsignals.sd from "
            "https://sourceforge.net/projects/nmrshiftdb2/files/data/ and put it there, "
            "or run the synthetic demo with configs/demo.yaml."
        )
    return load_nmrshiftdb(sdf, grid, min_peaks=int(d["min_peaks"]),
                           max_records=d.get("max_records"), progress=progress)
