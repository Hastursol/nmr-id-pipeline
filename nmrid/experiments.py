"""Experiment runner for RQ1-RQ4 and the open-set extension.

Trained models are cached under ``results/<run_name>/models`` and reused, so
RQ2 and RQ3 evaluate exactly the models reported in RQ1, and re-running an
experiment only retrains what is missing.  Every table is written as CSV (and
RQ1 also as a LaTeX table); every figure as a 300-dpi PNG.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import plots
from .augment import add_noise, shift_ppm
from .config import dataset_path, run_dir, save_config
from .dataset import TEST, TRAIN, VAL, NMRDataset
from .evaluate import (confusion_submatrix, cosine_scores, embed_cnn, identification_metrics,
                       most_confused, pair_metrics, scores_cnn, scores_rf, scores_siamese)
from .features import extract_features, feature_groups
from .models import count_parameters
from .train import get_device, load_torch, save_torch, train_cnn, train_rf, train_siamese

NAMES = {"cosine": "Cosine search (no learning)", "rf": "Random Forest (145 features)",
         "cnn": "1D CNN", "siamese": "Siamese CNN"}


class Runner:
    def __init__(self, cfg: dict, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.out = run_dir(cfg)
        for sub in ("models", "figures", "tables"):
            (self.out / sub).mkdir(exist_ok=True)
        path = dataset_path(cfg)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python -m nmrid prepare` first")
        self.ds = NMRDataset.load(path)
        self.device = get_device(cfg.get("device", "auto"))
        self.K = self.ds.n_classes
        self.top_k = tuple(cfg["experiments"]["top_k"])
        self._features: dict = {}
        save_config(cfg, self.out / "config.json")
        self.log(f"dataset: {self.ds.info}  device: {self.device}")

    # ------------------------------------------------------------ helpers
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def table(self, df: pd.DataFrame, name: str) -> Path:
        path = self.out / "tables" / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def fig(self, name: str) -> Path:
        return self.out / "figures" / f"{name}.png"

    def train_set(self, frac: float = 1.0):
        return self.ds.closed_set(TRAIN, frac, bool(self.cfg["data"]["include_reference_in_train"]))

    def feats(self, X: np.ndarray, key: str) -> np.ndarray:
        if key not in self._features:
            self._features[key] = extract_features(X, self.ds.grid)
        return self._features[key]

    def _meta_path(self, model_path: Path) -> Path:
        return model_path.with_suffix(model_path.suffix + ".json")

    def _write_meta(self, model_path: Path, **meta) -> None:
        with open(self._meta_path(model_path), "w") as fh:
            json.dump(meta, fh, indent=2)

    def meta(self, model_path: Path) -> dict:
        p = self._meta_path(model_path)
        return json.load(open(p)) if p.exists() else {}

    # --------------------------------------------------------- model cache
    def rf(self, seed: int = 0, frac: float = 1.0, kind: str = "features"):
        path = self.out / "models" / f"rf_{kind}_f{frac:g}_s{seed}.joblib"
        if path.exists():
            return joblib.load(path), path
        X, y = self.train_set(frac)
        if kind == "features":
            F = self.feats(X, f"train_f{frac:g}")
        elif kind == "raw":
            F = X
        elif kind == "cnn_embed":
            cnn, _ = self.cnn(seed)
            F = embed_cnn(cnn, X, self.device)
        else:
            raise ValueError(kind)
        self.log(f"training Random Forest ({kind}, fraction {frac:g}, seed {seed}) on {F.shape}")
        t0 = time.perf_counter()
        model = train_rf(F, y, self.cfg, seed)
        secs = time.perf_counter() - t0
        joblib.dump(model, path, compress=3)
        self._write_meta(path, train_seconds=secs, n_train=int(len(y)),
                         model_mb=round(path.stat().st_size / 2**20, 1))
        return model, path

    def cnn(self, seed: int = 0, frac: float = 1.0, tag: str = "", cfg: dict | None = None,
            classes: np.ndarray | None = None):
        cfg = cfg or self.cfg
        path = self.out / "models" / f"cnn{tag}_f{frac:g}_s{seed}.pt"
        if path.exists():
            model, _ = load_torch(path, self.device)
            return model, path
        X, y = self.train_set(frac)
        Xv, yv = self.ds.closed_set(VAL)
        if classes is not None:  # open-set: train only on the training compounds
            X, y = X[np.isin(y, classes)], y[np.isin(y, classes)]
            Xv, yv = Xv[np.isin(yv, classes)], yv[np.isin(yv, classes)]
        self.log(f"training 1D CNN{tag} (fraction {frac:g}, seed {seed}) on {X.shape}")
        t0 = time.perf_counter()
        model, hist = train_cnn(X, y, Xv, yv, self.K, cfg, self.device, seed, self.verbose)
        secs = time.perf_counter() - t0
        save_torch(model, path, "cnn", self.K, cfg, hist)
        self._write_meta(path, train_seconds=secs, n_train=int(len(y)), params=count_parameters(model),
                         best_epoch=hist["best_epoch"], epochs_run=len(hist["train_loss"]))
        plots.training_curves(hist, self.fig(f"training_cnn{tag}_f{frac:g}_s{seed}"), f"1D CNN{tag} training")
        return model, path

    def siamese(self, seed: int = 0, frac: float = 1.0, tag: str = "",
                allowed: np.ndarray | None = None, X_anchor=None, y_anchor=None, X_val=None, y_val=None):
        path = self.out / "models" / f"siamese{tag}_f{frac:g}_s{seed}.pt"
        if path.exists():
            model, _ = load_torch(path, self.device)
            return model, path
        if X_anchor is None:
            X_anchor, y_anchor = self.ds.closed_set(TRAIN, frac, include_reference=False)
            X_val, y_val = self.ds.closed_set(VAL)
        allowed = np.arange(self.K) if allowed is None else allowed
        self.log(f"training Siamese CNN{tag} (fraction {frac:g}, seed {seed}) on {X_anchor.shape} anchors")
        t0 = time.perf_counter()
        model, hist = train_siamese(X_anchor, y_anchor, self.ds.refs, allowed, X_val, y_val,
                                    self.cfg, self.device, seed, self.verbose)
        secs = time.perf_counter() - t0
        save_torch(model, path, "siamese", self.K, self.cfg, hist)
        self._write_meta(path, train_seconds=secs, n_anchor=int(len(y_anchor)),
                         params=count_parameters(model), best_epoch=hist["best_epoch"],
                         epochs_run=len(hist["train_loss"]))
        plots.training_curves(hist, self.fig(f"training_siamese{tag}_f{frac:g}_s{seed}"),
                              f"Siamese CNN{tag} training")
        return model, path

    def scores(self, model_key: str, X: np.ndarray, seed: int = 0, frac: float = 1.0,
               feat_key: str | None = None) -> tuple[np.ndarray, dict]:
        """Score matrix (N x K) of one model plus its training metadata."""
        if model_key == "cosine":
            return cosine_scores(X, self.ds.refs), {"train_seconds": 0.0, "params": 0}
        if model_key == "rf":
            model, path = self.rf(seed, frac)
            F = self.feats(X, feat_key) if feat_key else extract_features(X, self.ds.grid)
            return scores_rf(model, F, self.K), self.meta(path)
        if model_key == "cnn":
            model, path = self.cnn(seed, frac)
            return scores_cnn(model, X, self.device), self.meta(path)
        if model_key == "siamese":
            model, path = self.siamese(seed, frac)
            return scores_siamese(model, X, self.ds.refs, self.device, self.cfg["siamese"]["loss"]), self.meta(path)
        raise ValueError(model_key)

    # ----------------------------------------------------------------- RQ1
    def rq1(self) -> pd.DataFrame:
        self.log("\n=== RQ1: architecture comparison ===")
        X_test, y_test = self.ds.closed_set(TEST)
        has_real = len(self.ds.real_y) > 0
        rows = []
        for seed in self.cfg["experiments"]["seeds"]:
            for m in self.cfg["experiments"]["models"]:
                if m != "cosine" or seed == self.cfg["experiments"]["seeds"][0]:
                    self.scores(m, X_test[:1], seed)  # make sure the model exists before timing
                t0 = time.perf_counter()  # timing includes RF feature extraction
                S, meta = self.scores(m, X_test, seed)
                ms = 1000 * (time.perf_counter() - t0) / len(X_test)
                row = {"model": m, "seed": seed, **identification_metrics(S, y_test, self.K, self.top_k),
                       "ms_per_spectrum": ms, "train_seconds": meta.get("train_seconds", 0.0),
                       "params": meta.get("params", 0)}
                if has_real:
                    Sr, _ = self.scores(m, self.ds.real_X, seed, feat_key="real")
                    real = identification_metrics(Sr, self.ds.real_y, self.K, self.top_k)
                    row.update({f"real_{k}": v for k, v in real.items()})
                rows.append(row)
                if seed == self.cfg["experiments"]["seeds"][0]:
                    pred = S.argmax(1)
                    self.table(most_confused(y_test, pred, self.ds.names), f"rq1_most_confused_{m}")
                    cm, idx = confusion_submatrix(y_test, pred, self.K)
                    plots.confusion_heatmap(cm, [self.ds.names[i] for i in idx], self.fig(f"rq1_confusion_{m}"),
                                            f"{NAMES[m]} - most-confused compounds")
                self.log(f"  {m:8s} seed {seed}: top1 {row['top1']:.3f}  top5 {row.get('top5', float('nan')):.3f}  "
                         f"F1 {row['macro_f1']:.3f}  {ms:.2f} ms/spectrum")
        by_seed = pd.DataFrame(rows)
        self.table(by_seed, "rq1_results_by_seed")
        num = by_seed.drop(columns=["seed"]).groupby("model", sort=False)
        summary = num.mean().reset_index()
        if by_seed["seed"].nunique() > 1:
            std = num.std().reset_index()
            for c in ("top1", "top5", "macro_f1"):
                if c in std:
                    summary[c + "_std"] = std[c]
        self.table(summary, "rq1_results")
        self._latex_rq1(summary, has_real)
        plots.rq1_bars(summary, self.fig("rq1_comparison"))
        self._siamese_pair_metrics(X_test, y_test)
        self._example_figure()
        return summary

    def _latex_rq1(self, df: pd.DataFrame, has_real: bool) -> None:
        cols = ["Model", "Top-1", "Top-5", "Macro-F1", "ms/spectrum", "Parameters"]
        if has_real:
            cols.insert(4, "Top-1 (real)")
        lines = ["\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", "\\toprule",
                 " & ".join(cols) + " \\\\", "\\midrule"]
        for _, r in df.iterrows():
            def f(c):
                s = f"{r[c]:.3f}"
                return s + (f" $\\pm$ {r[c + '_std']:.3f}" if c + "_std" in r and not np.isnan(r[c + "_std"]) else "")
            cells = [NAMES.get(r["model"], r["model"]), f("top1"), f("top5"), f("macro_f1")]
            if has_real:
                cells.append(f"{r['real_top1']:.3f}")
            params = int(r["params"]) if r["params"] else 0
            cells += [f"{r['ms_per_spectrum']:.2f}", f"{params:,}" if params else "--"]
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (self.out / "tables" / "rq1_table.tex").write_text("\n".join(lines) + "\n")

    def _siamese_pair_metrics(self, X_test, y_test) -> None:
        """Wei et al. (2022) report pair ACC / TPR / FPR; the same numbers for this Siamese model."""
        if "siamese" not in self.cfg["experiments"]["models"] or self.cfg["siamese"]["loss"] != "bce":
            return
        model, _ = self.siamese(self.cfg["experiments"]["seeds"][0])
        rng = np.random.default_rng(7)
        neg = (y_test + rng.integers(1, self.K, len(y_test))) % self.K
        S = scores_siamese(model, X_test, self.ds.refs, self.device)
        probs = np.concatenate([S[np.arange(len(y_test)), y_test], S[np.arange(len(y_test)), neg]])
        same = np.concatenate([np.ones(len(y_test)), np.zeros(len(y_test))])
        res = pair_metrics(probs, same)
        self.table(pd.DataFrame([res]), "rq1_siamese_pair_metrics")
        self.log(f"  Siamese pair metrics (Wei et al. style): {res}")

    def _example_figure(self) -> None:
        c = int(self.ds.real_y[0]) if len(self.ds.real_y) else 0
        series = [("reference", self.ds.refs[c]),
                  ("simulated measurement", self.ds.X[np.flatnonzero(self.ds.y == c)[0]])]
        if len(self.ds.real_y):
            series.append(("real re-measurement", self.ds.real_X[0]))
        plots.example_spectra(self.ds.grid, series, self.fig("example_spectra"),
                              f"Example spectra - {str(self.ds.names[c])[:50]}")

    # ----------------------------------------------------------------- RQ2
    def rq2(self) -> pd.DataFrame:
        self.log("\n=== RQ2: robustness to noise and chemical-shift offsets ===")
        X_test, y_test = self.ds.closed_set(TEST)
        seed = self.cfg["experiments"]["seeds"][0]
        rows = []
        for sd in self.cfg["experiments"]["noise_levels"]:
            Xn = add_noise(X_test, float(sd), seed=1234)
            for m in self.cfg["experiments"]["models"]:
                S, _ = self.scores(m, Xn, seed)
                rows.append({"model": m, "noise_sd": sd, **identification_metrics(S, y_test, self.K, self.top_k)})
                self.log(f"  noise {sd:<5}  {m:8s} top1 {rows[-1]['top1']:.3f}")
        noise = pd.DataFrame(rows)
        self.table(noise, "rq2_noise")
        plots.rq2_curves(noise, "noise_sd", self.fig("rq2_noise"), "Gaussian noise standard deviation",
                         "RQ2 - accuracy under added noise")
        rows = []
        for d in self.cfg["experiments"]["shift_ppm"]:
            Xs = shift_ppm(X_test, float(d), self.ds.grid.width)
            for m in self.cfg["experiments"]["models"]:
                S, _ = self.scores(m, Xs, seed)
                rows.append({"model": m, "shift_ppm": d, **identification_metrics(S, y_test, self.K, self.top_k)})
            self.log(f"  shift {d:+.3f} ppm done")
        shift = pd.DataFrame(rows)
        self.table(shift, "rq2_shift")
        plots.rq2_curves(shift, "shift_ppm", self.fig("rq2_shift"), "Global chemical-shift offset (ppm)",
                         "RQ2 - accuracy under a chemical-shift offset")
        return noise

    # ----------------------------------------------------------------- RQ3
    def rq3(self) -> pd.DataFrame:
        self.log("\n=== RQ3: feature engineering vs deep learning ===")
        X_test, y_test = self.ds.closed_set(TEST)
        seed = self.cfg["experiments"]["seeds"][0]
        rows = []

        def add(key, label, S):
            rows.append({"variant": key, "label": label, **identification_metrics(S, y_test, self.K, self.top_k)})
            self.log(f"  {label:45s} top1 {rows[-1]['top1']:.3f}")

        rf_feat, _ = self.rf(seed, kind="features")
        add("rf_features", "RF on 145 hand-crafted features",
            scores_rf(rf_feat, self.feats(X_test, "test"), self.K))
        rf_raw, _ = self.rf(seed, kind="raw")
        add("rf_raw", "RF on the raw 2,000-bin spectrum", scores_rf(rf_raw, X_test, self.K))
        cnn, _ = self.cnn(seed)
        add("cnn", "1D CNN end-to-end", scores_cnn(cnn, X_test, self.device))
        rf_emb, _ = self.rf(seed, kind="cnn_embed")
        add("rf_cnn_embed", "RF on CNN-learned 256-d features",
            scores_rf(rf_emb, embed_cnn(cnn, X_test, self.device), self.K))
        gap_cfg = json.loads(json.dumps(self.cfg))
        gap_cfg["encoder"]["pool_bins"] = 1
        gap_cfg["encoder"]["bottleneck"] = None
        cnn_gap, _ = self.cnn(seed, tag="_gap", cfg=gap_cfg)
        add("cnn_gap", "1D CNN with global average pooling (ablation)", scores_cnn(cnn_gap, X_test, self.device))
        df = pd.DataFrame(rows)
        self.table(df, "rq3_results")
        plots.rq3_bars(df, self.fig("rq3_feature_engineering"))

        groups = feature_groups()
        imp = pd.Series(rf_feat.feature_importances_).groupby(groups).sum()
        imp = imp / imp.sum()
        self.table(imp.rename("importance").reset_index().rename(columns={"index": "group"}),
                   "rq3_rf_feature_group_importance")
        plots.feature_importance_groups(imp.to_dict(), self.fig("rq3_rf_feature_importance"))
        return df

    # ----------------------------------------------------------------- RQ4
    def rq4(self) -> pd.DataFrame:
        self.log("\n=== RQ4: learning curves ===")
        X_test, y_test = self.ds.closed_set(TEST)
        seed = self.cfg["experiments"]["seeds"][0]
        n_train = int(self.ds.info["replicate_split"][0])
        with_ref = int(bool(self.cfg["data"]["include_reference_in_train"]))
        rows = []
        for frac in self.cfg["experiments"]["train_fractions"]:
            frac = float(frac)
            per_compound = max(1, int(np.ceil(frac * n_train))) + with_ref
            for m in self.cfg["experiments"]["models"]:
                S, meta = self.scores(m, X_test, seed, frac)
                rows.append({"model": m, "train_fraction": frac, "train_spectra_per_compound": per_compound,
                             **identification_metrics(S, y_test, self.K, self.top_k),
                             "train_seconds": meta.get("train_seconds", 0.0)})
                self.log(f"  fraction {frac:.2f} ({per_compound}/compound)  {m:8s} top1 {rows[-1]['top1']:.3f}")
        df = pd.DataFrame(rows)
        self.table(df, "rq4_learning_curves")
        plots.rq4_curves(df, self.fig("rq4_learning_curves"))
        return df

    # ------------------------------------------------------------ open set
    def openset(self) -> pd.DataFrame:
        """Identify compounds never seen in training (compound-level 70/15/15 split)."""
        self.log("\n=== Open-set: compounds unseen during training ===")
        ds, seed = self.ds, self.cfg["experiments"]["seeds"][0]
        train_c, val_c = ds.compounds_in(TRAIN), ds.compounds_in(VAL)
        Xq, yq = ds.open_set(TEST)
        Xa, ya = ds.open_set(TRAIN)
        Xv, yv = ds.open_set(VAL)
        rows = []

        def add(key, S):
            rows.append({"model": key, **identification_metrics(S, yq, self.K, self.top_k)})
            self.log(f"  {key:10s} top1 {rows[-1]['top1']:.3f}  top5 {rows[-1].get('top5', float('nan')):.3f}")

        add("cosine", cosine_scores(Xq, ds.refs))
        sia, _ = self.siamese(seed, tag="_openset", allowed=train_c, X_anchor=Xa, y_anchor=ya, X_val=Xv, y_val=yv)
        add("siamese", scores_siamese(sia, Xq, ds.refs, self.device, self.cfg["siamese"]["loss"]))
        cnn, _ = self.cnn(seed, tag="_openset", classes=train_c)
        add("cnn_knn", cosine_scores(embed_cnn(cnn, Xq, self.device), embed_cnn(cnn, ds.refs, self.device)))
        df = pd.DataFrame(rows)
        df["n_train_compounds"], df["n_val_compounds"] = len(train_c), len(val_c)
        df["n_test_compounds"], df["gallery_size"] = int((ds.compound_split == TEST).sum()), self.K
        self.table(df, "openset_results")
        return df

    def run(self, which) -> None:
        steps = {"1": self.rq1, "2": self.rq2, "3": self.rq3, "4": self.rq4, "openset": self.openset}
        for w in (["1", "2", "3", "4", "openset"] if "all" in which else which):
            steps[str(w)]()
        self.log(f"\nresults written to {self.out.resolve()}")
