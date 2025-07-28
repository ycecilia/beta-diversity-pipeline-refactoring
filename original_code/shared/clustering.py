"""
Clustering algorithms for beta diversity analysis.

This module provides various clustering algorithms optimized for PCoA coordinates
from beta diversity analysis, including MeanShift and OPTICS.
"""

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, OPTICS
from sklearn.metrics import silhouette_score
from .logger import info


def generate_distinct_colors(n: int):
    """
    Generate distinct colors for clusters.

    Args:
        n: Number of colors to generate

    Returns:
        List of color strings in rgba format
    """
    from colormap import rgb2hex, hex2rgb

    base_palette = [
        "#ef476f",
        "#ffd166",
        "#06d6a0",
        "#118ab2",
        "#073b4c",
        "#f95738",
    ]

    if n <= len(base_palette):
        palette = base_palette[:n]
    else:
        palette = base_palette.copy()
        colors_needed = n - len(base_palette)

        for i in range(colors_needed):
            # Determine which two colors to interpolate between
            idx1, idx2 = (
                i % (len(base_palette) - 1),
                (i % (len(base_palette) - 1)) + 1,
            )

            # Convert hex to RGB
            rgb1 = hex2rgb(base_palette[idx1])
            rgb2 = hex2rgb(base_palette[idx2])

            # Interpolate
            t = (i // (len(base_palette) - 1)) / (
                colors_needed // (len(base_palette) - 1) + 1
            )
            new_rgb = tuple(int(rgb1[j] * (1 - t) + rgb2[j] * t) for j in range(3))

            # Convert back to hex and add to palette
            palette.append(rgb2hex(*new_rgb))

    # Convert to rgba
    return [
        f"rgba{tuple(int(c[i:i+2], 16) for i in (1, 3, 5)) + (1.0,)}" for c in palette
    ]


def cluster_with_meanshift(X):
    """
    Apply MeanShift clustering to PCoA coordinates.

    Args:
        X: numpy array of shape (n_samples, 2) containing PC1, PC2 coordinates

    Returns:
        tuple: (cluster_labels, unique_clusters, method_info)
    """
    bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=min(500, X.shape[0]))

    if bandwidth > 0:
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        unique_clusters = np.unique(ms.labels_).tolist()
        method_info = (
            f"MeanShift with bandwidth={bandwidth:.4f}, {len(unique_clusters)} clusters"
        )
        return ms.labels_, unique_clusters, method_info
    else:
        # Fallback if bandwidth estimation fails
        return np.zeros(X.shape[0]), [0], "MeanShift failed - single cluster"


def cluster_with_optics(X_or_D, *, distance=False):
    """
    Cluster samples with OPTICS and grid-search (min_samples, xi).
    X_or_D : array (nxp coordinates) OR (nxn distance matrix)
    distance: True if X_or_D is a pre-computed distance matrix
    Returns: labels, n_clusters, info
    """
    if distance:
        D = X_or_D
        n = D.shape[0]
        metric = "precomputed"
    else:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X_or_D)
        D = None
        n = X.shape[0]
        metric = "euclidean"

    min_samples_grid = [max(3, int(np.sqrt(n))),
                        int(np.sqrt(n)*1.5),
                        int(np.sqrt(n)*2)]
    xi_grid = [0.03, 0.05, 0.07, 0.10]

    best = (-np.inf, None, None, None)
    for k in min_samples_grid:
        for xi in xi_grid:
            optics = OPTICS(min_samples=k, xi=xi,
                            metric=metric, max_eps=np.inf)
            labels = optics.fit_predict(D if distance else X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue          # skip this (k, xi) combo

            mask = labels != -1
            if mask.sum() < 2:
                continue
            score = silhouette_score(
                D[mask][:, mask] if distance else
                (X if D is None else D)[mask],
                labels[mask],
                metric="precomputed" if distance else "euclidean"
            )
            best = max(best, (score, k, xi, labels))

    _, k_best, xi_best, labels = best
    clusters = np.unique(labels[labels != -1]).tolist()
    noise = (labels == -1).sum()
    info = f"OPTICS (min_samples={k_best}, xi={xi_best:.2f}) → {len(clusters)} clusters, {noise} noise"

    return labels, clusters, info


def cluster_with_dbscan_balanced(X):
    """
    Apply balanced DBSCAN clustering to PCoA coordinates.

    This is a balanced approach that aims for 3-6 meaningful clusters,
    falling back to K-means if DBSCAN doesn't find an appropriate number.

    Args:
        X: numpy array of shape (n_samples, 2) containing PC1, PC2 coordinates

    Returns:
        tuple: (cluster_labels, unique_clusters, method_info)
    """
    from sklearn.cluster import DBSCAN, KMeans

    # Try DBSCAN with moderate parameters first
    eps_moderate = np.std(X) * 0.15  # Moderate eps for balanced clusters
    min_samples_moderate = max(3, int(X.shape[0] * 0.02))  # 2% minimum or 3 points

    dbscan_moderate = DBSCAN(eps=eps_moderate, min_samples=min_samples_moderate)
    labels_moderate = dbscan_moderate.fit_predict(X)
    n_clusters_moderate = len(np.unique(labels_moderate[labels_moderate != -1]))

    # Use K-means with constrained k if DBSCAN doesn't find good clusters
    if n_clusters_moderate < 2 or n_clusters_moderate > 8:
        # Use silhouette analysis but constrain to 3-6 clusters for interpretability
        best_k = 3
        best_score = -1
        for k in range(3, min(7, X.shape[0] // 5)):  # Limit to 3-6 clusters
            if k >= X.shape[0]:
                break
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_k = k
                    best_score = score

        # Apply K-means with optimal k
        kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        unique_clusters = np.unique(cluster_labels).tolist()
        method_info = (
            f"Balanced K-means with k={best_k}, silhouette score={best_score:.3f}"
        )

        return cluster_labels, unique_clusters, method_info
    else:
        unique_clusters = np.unique(labels_moderate[labels_moderate != -1]).tolist()
        method_info = f"Balanced DBSCAN with {n_clusters_moderate} clusters, eps={eps_moderate:.4f}"

        return labels_moderate, unique_clusters, method_info


def apply_clustering(X, method="optics"):
    """
    Apply the specified clustering method to PCoA coordinates.

    Args:
        X: numpy array of shape (n_samples, 2) containing PC1, PC2 coordinates
        method: str, one of "meanshift", "optics", or "balanced"

    Returns:
        tuple: (cluster_labels, unique_clusters, colors, method_info)
    """
    if method == "meanshift":
        cluster_labels, unique_clusters, method_info = cluster_with_meanshift(X)
    elif method == "optics":
        cluster_labels, unique_clusters, method_info = cluster_with_optics(X)
    elif method == "balanced":
        cluster_labels, unique_clusters, method_info = cluster_with_dbscan_balanced(X)
    else:
        raise ValueError(
            f"Unknown clustering method: {method}. Available: meanshift, optics, balanced"
        )

    # Generate colors for clusters
    colors = generate_distinct_colors(len(unique_clusters)) if unique_clusters else []

    info(method_info)
    return cluster_labels, unique_clusters, colors, method_info
