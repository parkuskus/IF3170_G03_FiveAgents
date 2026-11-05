"""Shared preprocessing and small from-scratch ID3, C4.5, and CART baselines."""

from dataclasses import dataclass
from copy import copy, deepcopy
from itertools import combinations

import numpy as np
import pandas as pd


MISSING = "__MISSING__"
UNKNOWN = "__UNKNOWN__"


class SharedPreprocessor:
    """Median-impute numeric data and retain nominal categories as strings."""

    def fit(self, X):
        self.columns = list(X.columns)
        self.numeric = [c for c in self.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical = [c for c in self.columns if c not in self.numeric]
        self.numeric_fill = {
            c: X[c].median() if X[c].notna().any() else 0.0
            for c in self.numeric
        }
        self.categories = {
            c: set(X[c].dropna().astype(str).unique()) for c in self.categorical
        }
        return self

    def transform(self, X):
        result = X.reindex(columns=self.columns).copy()
        for col in self.numeric:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(self.numeric_fill[col])
        for col in self.categorical:
            values = result[col].astype("string")
            values = values.fillna(MISSING)
            result[col] = values.where(values.isin(self.categories[col]) | values.eq(MISSING), UNKNOWN).astype(object)
        return result

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def _entropy(zeros, ones):
    total = zeros + ones
    if total == 0:
        return 0.0
    p = zeros / total
    q = ones / total
    return -(p * np.log2(p) if p else 0.0) - (q * np.log2(q) if q else 0.0)


def _gini(zeros, ones):
    total = zeros + ones
    if total == 0:
        return 0.0
    p = zeros / total
    q = ones / total
    return 1.0 - p * p - q * q


@dataclass
class Node:
    prediction: int
    positive_probability: float
    n_samples: int
    feature: str | None = None
    kind: str | None = None
    threshold: float | None = None
    categories: frozenset | None = None
    branches: dict | None = None


class BaselineTree:
    """Greedy binary classifier using information gain, gain ratio, or Gini."""

    def __init__(self, criterion="gini", max_depth=8, min_samples_leaf=40, min_samples_split=80):
        if criterion not in {"information_gain", "gain_ratio", "gini"}:
            raise ValueError("criterion must be information_gain, gain_ratio, or gini")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.columns = list(X.columns)
        self.numeric = [c for c in self.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.values = {c: X[c].to_numpy() for c in self.columns}
        self.y = np.asarray(y, dtype=np.int8)
        if set(np.unique(self.y)) - {0, 1}:
            raise ValueError("BaselineTree expects binary labels 0/1")
        self.root = self._grow(np.arange(len(self.y)), 0)
        return self

    def _impurity(self, zero, one):
        return _gini(zero, one) if self.criterion == "gini" else _entropy(zero, one)

    def _score(self, groups, parent_impurity, parent_size):
        if len(groups) < 2 or any(len(group) < self.min_samples_leaf for group in groups):
            return None
        weighted = sum(len(group) * self._impurity(np.sum(self.y[group] == 0), np.sum(self.y[group] == 1)) for group in groups) / parent_size
        gain = parent_impurity - weighted
        if gain <= 1e-12:
            return None
        if self.criterion != "gain_ratio":
            return gain
        split_info = _entropy(0, 0)
        for group in groups:
            p = len(group) / parent_size
            split_info -= p * np.log2(p)
        return gain / split_info if split_info else None

    def _numeric_candidates(self, indices, col, parent_impurity):
        values = self.values[col][indices].astype(float)
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        if len(sorted_values) < 2:
            return
        cuts = np.flatnonzero(sorted_values[:-1] < sorted_values[1:]) + 1
        if len(cuts) > 64:
            cuts = np.unique(cuts[np.linspace(0, len(cuts) - 1, 64).astype(int)])
        for cut in cuts:
            if cut < self.min_samples_leaf or len(indices) - cut < self.min_samples_leaf:
                continue
            threshold = (sorted_values[cut - 1] + sorted_values[cut]) / 2
            left = indices[order[:cut]]
            right = indices[order[cut:]]
            score = self._score((left, right), parent_impurity, len(indices))
            if score is not None:
                yield score, col, "numeric", threshold, None, {"le": left, "gt": right}

    def _categorical_candidates(self, indices, col, parent_impurity):
        values = self.values[col][indices].astype(str)
        categories = np.unique(values)
        if len(categories) < 2:
            return
        if self.criterion != "gini":
            groups = {category: indices[values == category] for category in categories}
            score = self._score(tuple(groups.values()), parent_impurity, len(indices))
            if score is not None:
                yield score, col, "multiway", None, None, groups
            return

        # CART tests unique binary partitions. Above 10 levels, use one-vs-rest
        # candidates to avoid exponential work on high-cardinality features.
        if len(categories) <= 10:
            # Canonical partitions always contain the first level on the left.
            subsets = (frozenset([categories[0], *combo]) for size in range(len(categories)) for combo in combinations(categories[1:], size))
        else:
            subsets = (frozenset([category]) for category in categories)
        seen = set()
        for left_categories in subsets:
            if len(left_categories) == len(categories):
                continue
            key = tuple(sorted(left_categories))
            if key in seen:
                continue
            seen.add(key)
            left_mask = np.isin(values, list(left_categories))
            left, right = indices[left_mask], indices[~left_mask]
            score = self._score((left, right), parent_impurity, len(indices))
            if score is not None:
                yield score, col, "binary_category", None, left_categories, {"left": left, "right": right}

    def _grow(self, indices, depth):
        positives = int(self.y[indices].sum())
        negatives = len(indices) - positives
        majority = int(positives > negatives)
        node = Node(majority, positives / len(indices), len(indices))
        if (self.max_depth is not None and depth >= self.max_depth) or len(indices) < self.min_samples_split or positives == 0 or negatives == 0:
            return node
        parent_impurity = self._impurity(negatives, positives)
        best = None
        for col in self.columns:
            candidates = self._numeric_candidates(indices, col, parent_impurity) if col in self.numeric else self._categorical_candidates(indices, col, parent_impurity)
            for candidate in candidates:
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            return node
        _, node.feature, node.kind, node.threshold, node.categories, groups = best
        node.branches = {key: self._grow(group, depth + 1) for key, group in groups.items()}
        return node

    def _predict_one(self, row):
        node = self.root
        while node.branches is not None:
            value = row[node.feature]
            if node.kind == "numeric":
                key = "le" if float(value) <= node.threshold else "gt"
            elif node.kind == "binary_category":
                key = "left" if str(value) in node.categories else "right"
            else:
                key = str(value)
            child = node.branches.get(key)
            if child is None:
                return node.prediction
            node = child
        return node.prediction

    def predict(self, X):
        return [int(self._predict_one(row)) for row in X.to_dict(orient="records")]

    def _positive_probability_one(self, row):
        node = self.root
        while node.branches is not None:
            value = row[node.feature]
            if node.kind == "numeric":
                key = "le" if float(value) <= node.threshold else "gt"
            elif node.kind == "binary_category":
                key = "left" if str(value) in node.categories else "right"
            else:
                key = str(value)
            child = node.branches.get(key)
            if child is None:
                return node.positive_probability
            node = child
        return node.positive_probability

    def predict_proba(self, X):
        positive = np.asarray([
            self._positive_probability_one(row)
            for row in X.to_dict(orient="records")
        ])
        return np.column_stack((1.0 - positive, positive))

    def _prune_node(self, node, alpha, total_samples):
        leaf_error = node.n_samples * min(node.positive_probability, 1.0 - node.positive_probability) / total_samples
        if node.branches is None:
            return leaf_error, 1

        subtree_error = 0.0
        leaves = 0
        for child in node.branches.values():
            child_error, child_leaves = self._prune_node(child, alpha, total_samples)
            subtree_error += child_error
            leaves += child_leaves

        if leaf_error + alpha <= subtree_error + alpha * leaves:
            node.feature = node.kind = node.threshold = node.categories = None
            node.branches = None
            return leaf_error, 1
        return subtree_error, leaves

    def pruned(self, alpha):
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        result = copy(self)
        result.root = deepcopy(self.root)
        self._prune_node(result.root, alpha, len(self.y))
        return result

    def leaf_count(self):
        def count(node):
            return 1 if node.branches is None else sum(count(child) for child in node.branches.values())

        return count(self.root)
