"""Memory-compact Random Forest for problems with many classes.

scikit-learn stores a full class-probability vector at *every* node of every
tree.  With 1,000 compounds, 500 fully grown trees take about 6.4 GB of RAM
(measured on 15,000 training spectra: ~1,700 nodes x 1,000 classes x 8 bytes
per tree), which does not fit on a laptop next to everything else.

This class trains exactly the same trees - scikit-learn DecisionTreeClassifier
on a bootstrap sample (as sample weights, like RandomForestClassifier), sqrt(d)
candidate features per split, Gini impurity - and then keeps only what
prediction needs: the split structure and a sparse class distribution for each
leaf.  The forest's prediction is the average of the leaf distributions, as in
Breiman (2001) and scikit-learn.  The same 500-tree forest needs about 20 MB.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier


def _fit_tree(X, y, classes, max_depth, max_features, min_samples_leaf, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    weight = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
    tree = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf, random_state=int(seed % 2**31))
    tree.fit(X, y, sample_weight=weight)
    t = tree.tree_
    leaves = np.flatnonzero(t.children_left == -1)
    value = t.value[leaves, 0, :]  # class fractions of the leaves only
    value = value / np.maximum(value.sum(axis=1, keepdims=True), 1e-12)
    col = np.searchsorted(classes, tree.classes_)
    rows, cols = np.nonzero(value)
    dist = sparse.csr_matrix((value[rows, cols].astype(np.float32), (leaves[rows], col[cols])),
                             shape=(t.node_count, len(classes)))
    return {
        "left": t.children_left.astype(np.int32), "right": t.children_right.astype(np.int32),
        "feature": t.feature.astype(np.int32), "threshold": t.threshold.astype(np.float64),
        "dist": dist, "importance": tree.feature_importances_.astype(np.float64),
    }


class CompactRandomForest:
    def __init__(self, n_estimators=500, max_depth=30, max_features="sqrt", min_samples_leaf=1,
                 n_jobs=-1, random_state=0):
        self.n_estimators, self.max_depth = int(n_estimators), max_depth
        self.max_features, self.min_samples_leaf = max_features, int(min_samples_leaf)
        self.n_jobs, self.random_state = n_jobs, random_state

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.classes_ = np.unique(y)
        seeds = np.random.default_rng(self.random_state).integers(0, 2**62, self.n_estimators)
        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_tree)(X, y, self.classes_, self.max_depth, self.max_features,
                               self.min_samples_leaf, s) for s in seeds)
        imp = np.mean([t["importance"] for t in self.trees_], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        self.n_features_in_ = X.shape[1]
        return self

    @staticmethod
    def _apply(tree, X):
        node = np.zeros(len(X), dtype=np.int32)
        active = np.flatnonzero(tree["left"][node] != -1)
        while active.size:
            n = node[active]
            go_left = X[active, tree["feature"][n]] <= tree["threshold"][n]
            node[active] = np.where(go_left, tree["left"][n], tree["right"][n])
            active = active[tree["left"][node[active]] != -1]
        return node

    def predict_proba(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        out = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
        for tree in self.trees_:
            leaf = tree["dist"][self._apply(tree, X)].tocoo()
            out[leaf.row, leaf.col] += leaf.data  # (row, col) pairs are unique within a tree
        return out / len(self.trees_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
