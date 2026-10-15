"""Metric helpers shared by training (checkpoint selection) and evaluation."""
from __future__ import annotations

import math
from typing import Optional


def best_f1_threshold(labels: list[int], probs: list[float]) -> tuple[float, float]:
    best = (0.5, -1.0)
    for t in [i / 100 for i in range(5, 96)]:
        tp = sum(1 for y, p in zip(labels, probs) if y == 1 and p >= t)
        fp = sum(1 for y, p in zip(labels, probs) if y == 0 and p >= t)
        fn = sum(1 for y, p in zip(labels, probs) if y == 1 and p < t)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[1]:
            best = (t, f1)
    return best


def auroc(labels: list[int], probs: list[float]) -> Optional[float]:
    """Mann-Whitney AUROC with tie handling; None if only one class is present."""
    pos = sum(1 for y in labels if y == 1)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    ranked = sorted(enumerate(probs), key=lambda kv: kv[1])
    ranks = [0.0] * len(probs)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][1] == ranked[i][1]:
            j += 1
        for k in range(i, j + 1):
            ranks[ranked[k][0]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def ece(labels: list[int], probs: list[float], bins: int = 10) -> float:
    """Expected calibration error of the predicted class's confidence."""
    total, err = len(labels), 0.0
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for y, p in zip(labels, probs):
        conf = max(p, 1 - p)
        correct = int((p >= 0.5) == bool(y))
        buckets[min(bins - 1, int((conf - 0.5) * 2 * bins))].append((conf, correct))
    for b in buckets:
        if b:
            err += len(b) / total * abs(sum(c for c, _ in b) / len(b) - sum(k for _, k in b) / len(b))
    return err


def apply_temperature(probs: list[float], t: float) -> list[float]:
    out = []
    for p in probs:
        p = min(max(p, 1e-6), 1 - 1e-6)
        out.append(1 / (1 + math.exp(-math.log(p / (1 - p)) / t)))
    return out


def fit_temperature(labels: list[int], probs: list[float]) -> float:
    def nll(t: float) -> float:
        ps = apply_temperature(probs, t)
        return -sum(math.log(p) if y else math.log(1 - p) for y, p in zip(labels, ps)) / len(labels)
    return min((0.25 + 0.05 * i for i in range(96)), key=nll)


def log_loss(labels: list[int], probs: list[float], eps: float = 1e-6) -> float:
    return -sum(math.log(max(p, eps)) if y else math.log(max(1 - p, eps)) for y, p in zip(labels, probs)) / len(labels)
