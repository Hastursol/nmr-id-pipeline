"""Training loops for the three models, with early stopping and checkpoints.

Checkpoints are plain ``torch.save`` dictionaries holding the weights, the
constructor arguments, the training history and the configuration, so a model
can be rebuilt without the code that trained it (pSCNN saves a Keras .h5 plus
a separate pickle of the history; here both live in one file).
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from .augment import SpectrumAugmenter
from .evaluate import scores_siamese, topk_accuracy
from .rf import CompactRandomForest
from .models import SiameseNet, SpectrumCNN, contrastive_loss, encoder_kwargs_from_config


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


class EarlyStopping:
    """Keeps the best weights; stops after ``patience`` epochs without improvement."""

    def __init__(self, patience: int):
        self.patience, self.best, self.best_state, self.bad_epochs = patience, None, None, 0

    def step(self, score: tuple, model: nn.Module) -> bool:
        if self.best is None or score > self.best:
            self.best, self.bad_epochs = score, 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1
        return self.bad_epochs > self.patience


def _epoch_order(n: int, batch_size: int, min_batches: int, gen: torch.Generator | None = None,
                 rng: np.random.Generator | None = None) -> torch.Tensor:
    """Shuffled sample order for one epoch.

    Small training sets (the low end of the learning curves) are cycled until the
    epoch has at least ``min_batches`` batches, so every model gets a comparable
    number of optimisation steps; augmentation makes the repeats differ.
    """
    reps = max(1, int(np.ceil(min_batches * batch_size / max(1, n)))) if min_batches else 1
    if rng is not None:
        return torch.from_numpy(np.concatenate([rng.permutation(n) for _ in range(reps)]))
    return torch.cat([torch.randperm(n, generator=gen) for _ in range(reps)])


# ---------------------------------------------------------------------- CNN
def train_cnn(X_train, y_train, X_val, y_val, n_classes: int, cfg: dict, device,
              seed: int = 0, verbose: bool = True) -> tuple[SpectrumCNN, dict]:
    set_seed(seed)
    c = cfg["cnn"]
    model = SpectrumCNN(n_classes, encoder_kwargs_from_config(cfg), dropout=c["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    aug = SpectrumAugmenter.from_config(cfg)
    loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
    Xv, yv = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
    stopper = EarlyStopping(int(c["patience"]))
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds")}
    gen = torch.Generator().manual_seed(seed)
    bs = int(c["batch_size"])

    for epoch in range(int(c["epochs"])):
        t0 = time.perf_counter()
        model.train()
        perm = _epoch_order(len(Xt), bs, int(c.get("min_batches_per_epoch", 0)), gen)
        tot_loss, correct = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i: i + bs]
            xb, yb = aug(Xt[idx].to(device)), yt[idx].to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        n_seen = len(perm)
        model.eval()
        with torch.no_grad():
            v_loss, v_correct = 0.0, 0
            for i in range(0, len(Xv), 512):
                xb, yb = Xv[i: i + 512].to(device), yv[i: i + 512].to(device)
                logits = model(xb)
                v_loss += loss_fn(logits, yb).item() * len(xb)
                v_correct += (logits.argmax(1) == yb).sum().item()
        tr_loss, tr_acc = tot_loss / n_seen, correct / n_seen
        va_loss, va_acc = v_loss / max(1, len(Xv)), v_correct / max(1, len(Xv))
        sched.step(va_acc)
        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc, opt.param_groups[0]["lr"],
                               time.perf_counter() - t0)):
            hist[k].append(float(v))
        _log(f"  cnn epoch {epoch + 1:3d}  loss {tr_loss:.4f}  acc {tr_acc:.3f}  "
             f"val_loss {va_loss:.4f}  val_acc {va_acc:.3f}  ({hist['seconds'][-1]:.1f}s)", verbose)
        if stopper.step((va_acc, -va_loss), model):
            _log(f"  early stop after epoch {epoch + 1}", verbose)
            break
    model.load_state_dict(stopper.best_state)
    hist["best_epoch"] = int(np.argmax(hist["val_acc"])) + 1
    return model, hist


# ------------------------------------------------------------------ Siamese
class PairSampler:
    """Balanced positive / negative pairs, with hard negatives from look-alike compounds."""

    def __init__(self, y_anchor: np.ndarray, refs: np.ndarray, allowed: np.ndarray, cfg: dict,
                 rng: np.random.Generator):
        s = cfg["siamese"]
        self.rng = rng
        self.y = y_anchor
        self.allowed = np.asarray(allowed)
        self.ref_prob = float(s["ref_partner_prob"])
        self.hard_frac = float(s["hard_negative_frac"])
        self.members = {c: np.flatnonzero(y_anchor == c) for c in self.allowed}
        # hard negatives: the most similar references among the allowed compounds
        R = refs[self.allowed]
        Rn = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        sim = Rn @ Rn.T
        np.fill_diagonal(sim, -np.inf)
        pool = min(int(s["hard_pool"]), len(self.allowed) - 1)
        self.neighbours = {}
        for i, c in enumerate(self.allowed):
            nn_idx = np.argpartition(-sim[i], kth=pool - 1)[:pool] if pool > 0 else np.array([], int)
            self.neighbours[c] = self.allowed[nn_idx]

    def partner(self, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each class pick either its reference (-1) or one of its replicates (index)."""
        use_ref = self.rng.random(len(classes)) < self.ref_prob
        idx = np.full(len(classes), -1)
        for i, c in enumerate(classes):
            m = self.members.get(c)
            if not use_ref[i] and m is not None and len(m):
                idx[i] = m[self.rng.integers(len(m))]
        return idx, classes

    def negatives(self, classes: np.ndarray) -> np.ndarray:
        out = self.allowed[self.rng.integers(len(self.allowed), size=len(classes))]
        hard = self.rng.random(len(classes)) < self.hard_frac
        for i, c in enumerate(classes):
            nb = self.neighbours.get(c)
            if hard[i] and nb is not None and len(nb):
                out[i] = nb[self.rng.integers(len(nb))]
            while out[i] == c and len(self.allowed) > 1:
                out[i] = self.allowed[self.rng.integers(len(self.allowed))]
        return out


def _gather(X_anchor: torch.Tensor, refs: torch.Tensor, idx: np.ndarray, classes: np.ndarray):
    """Rows of the anchor set (idx >= 0) or of the reference set (idx == -1)."""
    out = refs[torch.from_numpy(classes)].clone()
    rep = idx >= 0
    if rep.any():
        out[torch.from_numpy(np.flatnonzero(rep))] = X_anchor[torch.from_numpy(idx[rep])]
    return out, torch.from_numpy(rep)


def train_siamese(X_anchor, y_anchor, refs, allowed, X_val, y_val, cfg: dict, device,
                  seed: int = 0, verbose: bool = True) -> tuple[SiameseNet, dict]:
    """Train on pairs built from ``X_anchor`` (replicates) and ``refs`` (one per compound).

    Only compounds in ``allowed`` are used, which is how the open-set protocol
    keeps its test compounds out of training.  Validation measures retrieval:
    top-1 accuracy of ``X_val`` queries against all references.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    s = cfg["siamese"]
    model = SiameseNet(encoder_kwargs_from_config(cfg), shared_weights=bool(s["shared_weights"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    aug = SpectrumAugmenter.from_config(cfg)
    bce = nn.BCEWithLogitsLoss()
    keep = np.isin(y_anchor, allowed)
    X_anchor, y_anchor = X_anchor[keep], y_anchor[keep]
    sampler = PairSampler(y_anchor, refs, allowed, cfg, rng)
    Xa, R = torch.from_numpy(X_anchor).float(), torch.from_numpy(refs).float()
    if len(X_val) > int(s["max_val_queries"]):
        sel = rng.choice(len(X_val), int(s["max_val_queries"]), replace=False)
        X_val, y_val = X_val[sel], y_val[sel]
    stopper = EarlyStopping(int(s["patience"]))
    hist = {k: [] for k in ("train_loss", "train_pair_acc", "val_top1", "lr", "seconds")}
    half = max(1, int(s["batch_size"]) // 2)

    for epoch in range(int(s["epochs"])):
        t0 = time.perf_counter()
        model.train()
        order = _epoch_order(len(Xa), half, int(s.get("min_batches_per_epoch", 0)), rng=rng).numpy()
        tot_loss, correct, count = 0.0, 0, 0
        for i in range(0, len(order), half):
            a_idx = order[i: i + half]
            cls = y_anchor[a_idx]
            p_idx, p_cls = sampler.partner(cls)
            n_idx, n_cls = sampler.partner(sampler.negatives(cls))
            query = Xa[torch.from_numpy(a_idx)].repeat(2, 1)
            pos, pos_rep = _gather(Xa, R, p_idx, p_cls)
            neg, neg_rep = _gather(Xa, R, n_idx, n_cls)
            ref = torch.cat([pos, neg])
            rep_mask = torch.cat([pos_rep, neg_rep]).float()[:, None]
            query = aug(query.to(device))
            ref = ref.to(device)
            ref = rep_mask.to(device) * aug(ref) + (1 - rep_mask.to(device)) * ref
            same = torch.cat([torch.ones(len(a_idx)), torch.zeros(len(a_idx))]).to(device)
            logit, eq, er = model(query, ref)
            if s["loss"] == "contrastive":
                loss = contrastive_loss(eq, er, same, float(s["margin"]))
                pred = (eq - er).norm(dim=-1) < float(s["margin"]) / 2
            else:
                loss = bce(logit, same)
                pred = logit > 0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(same)
            correct += (pred.float() == same).sum().item()
            count += len(same)
        val_scores = scores_siamese(model, X_val, refs, device, loss=s["loss"])
        val_top1 = topk_accuracy(val_scores, y_val, 1)
        sched.step(val_top1)
        for k, v in zip(hist, (tot_loss / count, correct / count, val_top1,
                               opt.param_groups[0]["lr"], time.perf_counter() - t0)):
            hist[k].append(float(v))
        _log(f"  siamese epoch {epoch + 1:3d}  loss {hist['train_loss'][-1]:.4f}  "
             f"pair_acc {hist['train_pair_acc'][-1]:.3f}  val_top1 {val_top1:.3f}  "
             f"({hist['seconds'][-1]:.1f}s)", verbose)
        if stopper.step((val_top1, -hist["train_loss"][-1]), model):
            _log(f"  early stop after epoch {epoch + 1}", verbose)
            break
    model.load_state_dict(stopper.best_state)
    hist["best_epoch"] = int(np.argmax(hist["val_top1"])) + 1
    return model, hist


# ------------------------------------------------------------ Random Forest
def train_rf(F_train: np.ndarray, y_train: np.ndarray, cfg: dict, seed: int = 0):
    """Random Forest; ``rf.backend: compact`` (default) keeps memory low for many classes."""
    r = cfg["rf"]
    if r.get("backend", "compact") == "compact":
        return CompactRandomForest(
            n_estimators=int(r["n_estimators"]), max_depth=r["max_depth"],
            max_features=r["max_features"], min_samples_leaf=int(r["min_samples_leaf"]),
            n_jobs=int(r["n_jobs"]), random_state=seed,
        ).fit(F_train, y_train)
    rf = RandomForestClassifier(
        n_estimators=int(r["n_estimators"]), max_depth=r["max_depth"],
        max_features=r["max_features"], min_samples_leaf=int(r["min_samples_leaf"]),
        n_jobs=int(r["n_jobs"]), random_state=seed,
    )
    return rf.fit(F_train, y_train)


# -------------------------------------------------------------- persistence
def save_torch(model: nn.Module, path: str | Path, kind: str, n_classes: int, cfg: dict, hist: dict) -> None:
    torch.save({"kind": kind, "state_dict": model.state_dict(), "n_classes": n_classes,
                "config": cfg, "history": hist}, path)


def load_torch(path: str | Path, device) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    if ck["kind"] == "cnn":
        model = SpectrumCNN(ck["n_classes"], encoder_kwargs_from_config(cfg), dropout=cfg["cnn"]["dropout"])
    else:
        model = SiameseNet(encoder_kwargs_from_config(cfg), shared_weights=bool(cfg["siamese"]["shared_weights"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
