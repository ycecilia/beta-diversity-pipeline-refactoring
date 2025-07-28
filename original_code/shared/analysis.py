"""
Analysis helper functions for beta diversity calculations.

This module contains utilities for statistical analysis, including
optimal clustering analysis and tick position calculations.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def best_tick_count(values, k_min: int = 3, k_max: int = 7) -> int:
    """
    Pick an optimal k in [k_min, k_max] using the silhouette score.
    Falls back to the largest feasible k if there are too few unique points.

    Args:
        values: Array-like values to analyze
        k_min: Minimum number of clusters to consider
        k_max: Maximum number of clusters to consider

    Returns:
        Optimal number of clusters
    """
    values = np.asarray(values).reshape(-1, 1)
    uniq = len(np.unique(values))

    # shrink the allowed window if the data can't support many clusters
    k_max = min(k_max, uniq)
    k_min = min(k_min, k_max)

    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto").fit(values)
        score = silhouette_score(values, km.labels_)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def ticks_from_kmeans(
    values, *, k: int | None = None, k_min: int = 3, k_max: int = 7
) -> tuple[list[float], list[str]]:
    """
    Return tick positions/text constrained to [k_min, k_max] clusters.

    Args:
        values: Array-like values to create ticks for
        k: Specific number of clusters (if None, will be optimized)
        k_min: Minimum number of clusters to consider
        k_max: Maximum number of clusters to consider

    Returns:
        Tuple of (tick_positions, tick_labels)
    """
    if k is None:
        k = best_tick_count(values, k_min, k_max)

    km = KMeans(n_clusters=k, n_init="auto").fit(np.asarray(values).reshape(-1, 1))
    centres = sorted(km.cluster_centers_.ravel())
    labels = [f"{c:.2f}" for c in centres]
    return centres, labels
