"""Scoring and metrics.

Every model is reduced to the same interface: a score matrix of shape
(n_queries, n_classes) in which a larger value means "more likely this
compound".  Top-k accuracy, macro-F1 and confusion analysis are then computed
identically for all models, which is what makes the RQ1 comparison fair.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


# ------------------------------------------------------------------ scoring
def _batches(n: int, size: int):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


@torch.no_grad()
def scores_cnn(model, X: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for sl in _batches(len(X), batch_size):
        logits = model(torch.from_numpy(np.ascontiguousarray(X[sl])).float().to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


@torch.no_grad()
def embed_cnn(model, X: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = [model.embed(torch.from_numpy(np.ascontiguousarray(X[sl])).float().to(device)).cpu().numpy()
           for sl in _batches(len(X), batch_size)]
    return np.concatenate(out)


@torch.no_grad()
def scores_siamese(model, X: np.ndarray, gallery: np.ndarray, device, loss: str = "bce",
                   batch_size: int = 32) -> np.ndarray:
    """Score every query against every gallery (reference) spectrum.

    With the BCE head the score is P(same compound); with the contrastive loss
    it is the negative embedding distance.  Gallery embeddings are computed once.
    """
    model.eval()
    G = torch.cat([model.embed_ref(torch.from_numpy(np.ascontiguousarray(gallery[sl])).float().to(device))
                   for sl in _batches(len(gallery), 256)])
    out = []
    for sl in _batches(len(X), batch_size):
        q = model.embed_query(torch.from_numpy(np.ascontiguousarray(X[sl])).float().to(device))
        if loss == "contrastive":
            s = -torch.cdist(q, G)
        else:
            b, k, d = q.shape[0], G.shape[0], q.shape[1]
            qe = q[:, None, :].expand(b, k, d)
            ge = G[None, :, :].expand(b, k, d)
            s = torch.sigmoid(model.pair_logit(qe.reshape(-1, d), ge.reshape(-1, d))).view(b, k)
        out.append(s.cpu().numpy())
    return np.concatenate(out)


def scores_rf(rf, F: np.ndarray, n_classes: int) -> np.ndarray:
    proba = rf.predict_proba(F)
    out = np.zeros((len(F), n_classes), np.float32)
    out[:, rf.classes_.astype(int)] = proba
    return out


def cosine_scores(X: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """No-learning baseline: cosine similarity between the query and each reference."""
    a = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    b = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ------------------------------------------------------------------ metrics
def topk_accuracy(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return float("nan")
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(np.any(top == y[:, None], axis=1)))


def identification_metrics(scores: np.ndarray, y: np.ndarray, n_classes: int,
                           top_k=(1, 5), class_subset: np.ndarray | None = None) -> dict:
    """Top-k accuracy plus macro precision / recall / F1 over the classes present in ``y``."""
    pred = scores.argmax(axis=1)
    labels = np.unique(y) if class_subset is None else class_subset
    out = {f"top{k}": topk_accuracy(scores, y, k) for k in top_k}
    out["macro_f1"] = float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))
    out["macro_precision"] = float(precision_score(y, pred, labels=labels, average="macro", zero_division=0))
    out["macro_recall"] = float(recall_score(y, pred, labels=labels, average="macro", zero_division=0))
    out["n_queries"] = int(len(y))
    return out


def pair_metrics(prob: np.ndarray, same: np.ndarray, threshold: float = 0.5) -> dict:
    """ACC / TPR / FPR of same-compound decisions, the metrics reported by Wei et al. (2022)."""
    pred = prob >= threshold
    same = same.astype(bool)
    tp, fn = np.sum(pred & same), np.sum(~pred & same)
    fp, tn = np.sum(pred & ~same), np.sum(~pred & ~same)
    return {"pair_acc": float((tp + tn) / max(1, len(same))),
            "pair_tpr": float(tp / max(1, tp + fn)),
            "pair_fpr": float(fp / max(1, fp + tn)),
            "n_pairs": int(len(same))}


def most_confused(y: np.ndarray, pred: np.ndarray, names: list, n: int = 20) -> pd.DataFrame:
    labels = np.unique(np.concatenate([y, pred]))
    cm = confusion_matrix(y, pred, labels=labels)
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:n]
    rows = []
    for f in flat:
        i, j = np.unravel_index(f, cm.shape)
        if cm[i, j] == 0:
            break
        rows.append({"true": names[labels[i]], "predicted": names[labels[j]], "count": int(cm[i, j]),
                     "true_idx": int(labels[i]), "pred_idx": int(labels[j])})
    return pd.DataFrame(rows)


def confusion_submatrix(y: np.ndarray, pred: np.ndarray, n_classes: int, n_show: int = 25):
    """Confusion matrix restricted to the ``n_show`` classes involved in the most errors."""
    cm = confusion_matrix(y, pred, labels=np.arange(n_classes))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    involvement = off.sum(axis=0) + off.sum(axis=1)
    idx = np.argsort(involvement)[::-1][:n_show]
    idx = idx[involvement[idx] > 0]
    if len(idx) == 0:
        idx = np.arange(min(n_show, n_classes))
    return cm[np.ix_(idx, idx)], idx
