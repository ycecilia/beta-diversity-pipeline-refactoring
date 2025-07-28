"""
Clustering analysis module for beta diversity analysis.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import MeanShift, OPTICS, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

from .config import get_config
from .exceptions import ClusteringError
from .logging_config import get_logger, performance_tracker


class ClusteringAnalyzer:
    """Comprehensive clustering analysis for beta diversity data."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        self.color_palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17bec",
            "#aec7e8",
            "#ffbb78",
            "#98df8a",
            "#ff9896",
            "#c5b0d5",
        ]

    @performance_tracker("apply_clustering")
    def apply_clustering(
        self, coordinates: np.ndarray, method: str = "meanshift", **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """
        Apply clustering to PCoA coordinates.

        Args:
            coordinates: PCoA coordinates matrix (n_samples x n_dimensions)
            method: Clustering method to use
            **kwargs: Additional parameters for clustering method

        Returns:
            Tuple of (cluster_labels, unique_clusters, colors, method_info)
        """
        try:
            self.logger.info(
                f"Applying {method} clustering to {coordinates.shape[0]} samples"
            )

            # Validate inputs
            if coordinates.shape[0] < 3:
                raise ClusteringError("Need at least 3 samples for clustering")

            if coordinates.shape[1] < 2:
                raise ClusteringError("Need at least 2 dimensions for clustering")

            # Always use the standard MeanShift implementation for exact matching
            # Apply clustering based on method
            if method.lower() == "meanshift":
                cluster_labels, method_info = self._apply_meanshift(
                    coordinates, **kwargs
                )
            elif method.lower() == "optics":
                cluster_labels, method_info = self._apply_optics(coordinates, **kwargs)
            elif method.lower() == "balanced":
                cluster_labels, method_info = self._apply_balanced_clustering(
                    coordinates, **kwargs
                )
            elif method.lower() == "kmeans":
                cluster_labels, method_info = self._apply_kmeans(coordinates, **kwargs)
            elif method.lower() == "dbscan":
                cluster_labels, method_info = self._apply_dbscan(coordinates, **kwargs)
            else:
                raise ClusteringError(f"Unsupported clustering method: {method}")

            # Get unique clusters (excluding noise if present)
            unique_clusters = np.unique(cluster_labels)
            if -1 in unique_clusters:  # Remove noise cluster
                unique_clusters = unique_clusters[unique_clusters != -1]

            # Generate colors for clusters
            colors = self._generate_cluster_colors(len(unique_clusters))

            # Calculate clustering metrics
            method_info.update(
                self._calculate_clustering_metrics(coordinates, cluster_labels)
            )

            silhouette_score = method_info.get("silhouette_score", None)
            silhouette_str = (
                f"{silhouette_score:.3f}" if silhouette_score is not None else "N/A"
            )

            self.logger.info(
                f"Clustering completed: {len(unique_clusters)} clusters found "
                f"(silhouette score: {silhouette_str})"
            )

            return cluster_labels, unique_clusters, colors, method_info

        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            raise ClusteringError(f"Clustering failed: {e}")

    def _apply_meanshift(
        self, coordinates: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply Mean Shift clustering - EXACT match to original implementation."""

        # Estimate bandwidth using EXACT same parameters as original
        from sklearn.cluster import estimate_bandwidth

        bandwidth = estimate_bandwidth(
            coordinates, quantile=0.1, n_samples=min(500, coordinates.shape[0])
        )

        if bandwidth > 0:
            # Use EXACT same parameters as original
            clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            clusterer.fit(coordinates)
            cluster_labels = clusterer.labels_

            method_info = {
                "method": "meanshift",
                "bandwidth": bandwidth,
                "n_clusters": len(np.unique(cluster_labels)),
                "cluster_centers": (
                    clusterer.cluster_centers_.tolist()
                    if hasattr(clusterer, "cluster_centers_")
                    else []
                ),
            }
        else:
            # Fallback if bandwidth estimation fails (match original exactly)
            cluster_labels = np.zeros(coordinates.shape[0])
            method_info = {
                "method": "meanshift",
                "bandwidth": 0,
                "n_clusters": 1,
                "cluster_centers": [],
                "note": "MeanShift failed - single cluster",
            }

        return cluster_labels, method_info

    def _apply_optics(
        self, coordinates: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply OPTICS clustering."""
        # Default parameters
        min_samples = kwargs.get("min_samples", max(2, coordinates.shape[0] // 10))
        xi = kwargs.get("xi", 0.1)
        min_cluster_size = kwargs.get(
            "min_cluster_size", max(2, coordinates.shape[0] // 20)
        )

        clusterer = OPTICS(
            min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, n_jobs=-1
        )

        cluster_labels = clusterer.fit_predict(coordinates)

        method_info = {
            "method": "optics",
            "min_samples": min_samples,
            "xi": xi,
            "min_cluster_size": min_cluster_size,
            "n_clusters": len(np.unique(cluster_labels[cluster_labels != -1])),
        }

        return cluster_labels, method_info

    def _apply_balanced_clustering(
        self, coordinates: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply balanced clustering approach (combines multiple methods)."""
        n_samples = coordinates.shape[0]

        # Try different clustering methods and choose the best one
        methods_to_try = []

        # Add K-means with different k values
        for k in range(2, min(8, n_samples // 2)):
            methods_to_try.append(("kmeans", {"n_clusters": k}))

        # Add DBSCAN with different parameters
        if n_samples >= 5:
            for eps in [0.5, 1.0, 1.5]:
                for min_samples in [2, 3]:
                    methods_to_try.append(
                        ("dbscan", {"eps": eps, "min_samples": min_samples})
                    )

        # Add Mean Shift
        methods_to_try.append(("meanshift", {}))

        best_score = -1
        best_labels = None
        best_method_info = None

        for method, params in methods_to_try:
            try:
                if method == "kmeans":
                    labels, info = self._apply_kmeans(coordinates, **params)
                elif method == "dbscan":
                    labels, info = self._apply_dbscan(coordinates, **params)
                elif method == "meanshift":
                    labels, info = self._apply_meanshift(coordinates, **params)
                else:
                    continue

                # Skip if only one cluster or all noise
                unique_labels = np.unique(labels)
                if len(unique_labels) < 2 or (
                    len(unique_labels) == 2 and -1 in unique_labels
                ):
                    continue

                # Calculate silhouette score
                if len(unique_labels) >= 2 and len(unique_labels) < n_samples:
                    score = silhouette_score(coordinates, labels)
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_method_info = info
                        best_method_info["selection_score"] = score

            except Exception as e:
                self.logger.debug(f"Method {method} with params {params} failed: {e}")
                continue

        # Fallback to simple k-means if no good clustering found
        if best_labels is None:
            k = min(3, n_samples // 2)
            best_labels, best_method_info = self._apply_kmeans(
                coordinates, n_clusters=k
            )
            best_method_info["method"] = "balanced_fallback"
        else:
            best_method_info["method"] = "balanced"

        return best_labels, best_method_info

    def _apply_kmeans(
        self, coordinates: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply K-means clustering with performance optimizations."""
        n_clusters = kwargs.get("n_clusters", min(3, coordinates.shape[0] // 2))
        random_state = kwargs.get("random_state", 42)

        # Optimize for fast mode
        if hasattr(self.config, "analysis") and self.config.analysis.fast_mode:
            n_init = 3  # Fewer initializations for speed
            max_iter = 100  # Fewer iterations for speed
        else:
            n_init = 10
            max_iter = 300

        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )

        cluster_labels = clusterer.fit_predict(coordinates)

        method_info = {
            "method": "kmeans",
            "n_clusters": n_clusters,
            "inertia": float(clusterer.inertia_),
            "n_init": n_init,
            "max_iter": max_iter,
            "cluster_centers": clusterer.cluster_centers_.tolist(),
        }

        return cluster_labels, method_info

    def _apply_dbscan(
        self, coordinates: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply DBSCAN clustering."""
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", max(2, coordinates.shape[0] // 10))

        clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)

        cluster_labels = clusterer.fit_predict(coordinates)

        method_info = {
            "method": "dbscan",
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": len(np.unique(cluster_labels[cluster_labels != -1])),
            "n_noise": np.sum(cluster_labels == -1),
        }

        return cluster_labels, method_info

    def _generate_cluster_colors(self, n_clusters: int) -> List[str]:
        """Generate colors for clusters."""
        if n_clusters <= len(self.color_palette):
            return self.color_palette[:n_clusters]
        else:
            # Generate additional colors if needed
            import matplotlib.colors as mcolors

            additional_colors = []
            for i in range(n_clusters - len(self.color_palette)):
                # Generate colors using HSV space
                hue = (i * 137.508) % 360  # Golden angle approximation
                color = mcolors.hsv_to_rgb([hue / 360, 0.7, 0.9])
                hex_color = mcolors.to_hex(color)
                additional_colors.append(hex_color)

            return self.color_palette + additional_colors

    def _calculate_clustering_metrics(
        self, coordinates: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        metrics = {}

        try:
            # Filter out noise points for metrics calculation
            mask = labels != -1
            if np.sum(mask) < 2:
                return metrics

            filtered_coords = coordinates[mask]
            filtered_labels = labels[mask]

            unique_labels = np.unique(filtered_labels)
            n_clusters = len(unique_labels)

            if n_clusters >= 2 and n_clusters < len(filtered_labels):
                # Silhouette score
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    metrics["silhouette_score"] = float(
                        silhouette_score(filtered_coords, filtered_labels)
                    )

                # Calinski-Harabasz score
                metrics["calinski_harabasz_score"] = float(
                    calinski_harabasz_score(filtered_coords, filtered_labels)
                )

            # Basic statistics
            metrics["n_clusters"] = n_clusters
            metrics["n_noise_points"] = int(np.sum(labels == -1))
            metrics["noise_ratio"] = float(np.sum(labels == -1) / len(labels))

        except Exception as e:
            self.logger.debug(f"Failed to calculate clustering metrics: {e}")

        return metrics

    def optimize_clustering_parameters(
        self,
        coordinates: np.ndarray,
        method: str = "kmeans",
        param_ranges: Optional[Dict[str, List]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize clustering parameters using grid search.

        Args:
            coordinates: PCoA coordinates matrix
            method: Clustering method
            param_ranges: Parameter ranges to search

        Returns:
            Best parameters and scores
        """
        try:
            self.logger.info(f"Optimizing parameters for {method} clustering")

            if param_ranges is None:
                param_ranges = self._get_default_param_ranges(coordinates, method)

            best_score = -1
            best_params = None
            best_labels = None
            results = []

            # Grid search over parameter combinations
            param_combinations = self._generate_param_combinations(param_ranges)

            for params in param_combinations:
                try:
                    if method.lower() == "kmeans":
                        labels, info = self._apply_kmeans(coordinates, **params)
                    elif method.lower() == "dbscan":
                        labels, info = self._apply_dbscan(coordinates, **params)
                    elif method.lower() == "meanshift":
                        labels, info = self._apply_meanshift(coordinates, **params)
                    else:
                        continue

                    # Calculate metrics
                    metrics = self._calculate_clustering_metrics(coordinates, labels)

                    # Use silhouette score as primary metric
                    score = metrics.get("silhouette_score", -1)

                    results.append(
                        {
                            "params": params,
                            "score": score,
                            "metrics": metrics,
                            "labels": labels,
                        }
                    )

                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_labels = labels

                except Exception as e:
                    self.logger.debug(f"Parameter combination {params} failed: {e}")
                    continue

            optimization_results = {
                "best_params": best_params,
                "best_score": best_score,
                "best_labels": best_labels,
                "all_results": results,
                "method": method,
            }

            self.logger.info(
                f"Parameter optimization completed: best score = {best_score:.3f}, "
                f"best params = {best_params}"
            )

            return optimization_results

        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
            raise ClusteringError(f"Parameter optimization failed: {e}")

    def _get_default_param_ranges(
        self, coordinates: np.ndarray, method: str
    ) -> Dict[str, List]:
        """Get default parameter ranges for optimization."""
        n_samples = coordinates.shape[0]

        if method.lower() == "kmeans":
            return {"n_clusters": list(range(2, min(8, n_samples // 2)))}
        elif method.lower() == "dbscan":
            return {
                "eps": [0.3, 0.5, 0.7, 1.0, 1.5],
                "min_samples": [2, 3, 5, max(2, n_samples // 20)],
            }
        elif method.lower() == "meanshift":
            # Estimate bandwidth range
            from sklearn.cluster import estimate_bandwidth

            base_bw = estimate_bandwidth(coordinates, quantile=0.3)
            if base_bw <= 0:
                base_bw = 1.0

            return {"bandwidth": [base_bw * 0.5, base_bw, base_bw * 1.5, base_bw * 2.0]}
        else:
            return {}

    def _generate_param_combinations(
        self, param_ranges: Dict[str, List]
    ) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        from itertools import product

        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        combinations = []
        for combo in product(*param_values):
            combinations.append(dict(zip(param_names, combo)))

        return combinations

    def analyze_cluster_characteristics(
        self,
        coordinates: np.ndarray,
        cluster_labels: np.ndarray,
        metadata: Optional[pl.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Analyze characteristics of identified clusters.

        Args:
            coordinates: PCoA coordinates
            cluster_labels: Cluster assignments
            metadata: Optional metadata for additional analysis

        Returns:
            Cluster characteristics analysis
        """
        try:
            self.logger.info("Analyzing cluster characteristics")

            unique_clusters = np.unique(cluster_labels)
            if -1 in unique_clusters:
                unique_clusters = unique_clusters[unique_clusters != -1]

            cluster_analysis = {
                "n_clusters": len(unique_clusters),
                "cluster_stats": {},
                "cluster_separation": {},
                "metadata_associations": {},
            }

            # Analyze each cluster
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                cluster_coords = coordinates[mask]

                # Basic statistics
                stats = {
                    "size": int(np.sum(mask)),
                    "centroid": cluster_coords.mean(axis=0).tolist(),
                    "std": cluster_coords.std(axis=0).tolist(),
                    "diameter": float(
                        np.max(
                            np.linalg.norm(
                                cluster_coords[:, np.newaxis] - cluster_coords, axis=2
                            )
                        )
                    ),
                }

                cluster_analysis["cluster_stats"][str(cluster_id)] = stats

            # Calculate cluster separation
            centroids = np.array(
                [
                    cluster_analysis["cluster_stats"][str(cid)]["centroid"]
                    for cid in unique_clusters
                ]
            )

            if len(centroids) > 1:
                # Calculate pairwise distances between centroids
                from scipy.spatial.distance import pdist, squareform

                centroid_distances = squareform(pdist(centroids))

                cluster_analysis["cluster_separation"] = {
                    "mean_distance": float(
                        np.mean(centroid_distances[centroid_distances > 0])
                    ),
                    "min_distance": float(
                        np.min(centroid_distances[centroid_distances > 0])
                    ),
                    "max_distance": float(np.max(centroid_distances)),
                }

            # Analyze metadata associations if provided
            if metadata is not None:
                cluster_analysis["metadata_associations"] = (
                    self._analyze_metadata_associations(cluster_labels, metadata)
                )

            return cluster_analysis

        except Exception as e:
            self.logger.error(f"Cluster characteristics analysis failed: {e}")
            raise ClusteringError(f"Cluster characteristics analysis failed: {e}")

    def _analyze_metadata_associations(
        self, cluster_labels: np.ndarray, metadata: pl.DataFrame
    ) -> Dict[str, Any]:
        """Analyze associations between clusters and metadata variables."""
        associations = {}

        try:
            # Add cluster labels to metadata
            metadata_with_clusters = metadata.with_columns(
                pl.Series("cluster", cluster_labels)
            )

            # Analyze categorical variables
            categorical_cols = []
            for col in metadata.columns:
                if col not in [
                    "sample_id",
                    "latitude",
                    "longitude",
                ] and metadata.schema[col] in [pl.Utf8, pl.Categorical]:
                    categorical_cols.append(col)

            for col in categorical_cols:
                # Calculate association using contingency table
                contingency = (
                    metadata_with_clusters.group_by([col, "cluster"])
                    .agg(pl.count().alias("count"))
                    .pivot(values="count", index=col, on="cluster")
                    .fill_null(0)
                )

                associations[col] = {
                    "type": "categorical",
                    "contingency_table": contingency.to_dicts(),
                }

            # Analyze numerical variables
            numerical_cols = []
            for col in metadata.columns:
                if metadata.schema[col] in [pl.Int64, pl.Float64]:
                    numerical_cols.append(col)

            for col in numerical_cols:
                cluster_means = metadata_with_clusters.group_by("cluster").agg(
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).std().alias("std"),
                    pl.col(col).count().alias("count"),
                )

                associations[col] = {
                    "type": "numerical",
                    "cluster_statistics": cluster_means.to_dicts(),
                }

        except Exception as e:
            self.logger.debug(f"Metadata association analysis failed: {e}")

        return associations
