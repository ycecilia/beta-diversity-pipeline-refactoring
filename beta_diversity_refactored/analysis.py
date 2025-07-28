"""
Beta diversity analysis module.
"""

import numpy as np
import polars as pl
import hashlib
import pickle
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from skbio import DistanceMatrix, diversity
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa
from pathlib import Path

from .config import get_config
from .exceptions import AnalysisError, InsufficientDataError
from .logging_config import get_logger, performance_tracker
from .validation import DataValidator


@dataclass
class BetaDiversityResults:
    """Results from beta diversity analysis."""

    distance_matrix: DistanceMatrix
    pcoa_results: Any
    permanova_results: Optional[Dict[str, Any]]
    variance_explained: Dict[str, float]
    sample_scores: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class PCoAResults:
    """PCoA analysis results."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    proportion_explained: np.ndarray
    sample_scores: Dict[str, np.ndarray]
    total_variance: float


class BetaDiversityAnalyzer:
    """Comprehensive beta diversity analysis."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        self.validator = DataValidator(config)

        # Initialize cache directory
        if (
            hasattr(self.config, "analysis")
            and hasattr(self.config.analysis, "enable_caching")
            and self.config.analysis.enable_caching
        ):
            self.cache_dir = Path(self.config.storage.cache_dir) / "analysis"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def _get_cache_key(
        self, data_hash: str, operation: str, params: dict = None
    ) -> str:
        """Generate cache key for operation."""
        key_parts = [data_hash, operation]
        if params:
            param_str = str(sorted(params.items()))
            key_parts.append(hashlib.md5(param_str.encode()).hexdigest()[:8])
        return "_".join(key_parts)

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve result from cache."""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load from cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, result: Any) -> None:
        """Save result to cache."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")

    def _get_data_hash(self, otu_matrix: pl.DataFrame, metric: str) -> str:
        """Generate hash for OTU matrix and parameters."""
        # Create a hash of the matrix data and parameters
        matrix_data = otu_matrix.to_numpy().tobytes()
        param_str = f"{metric}_{otu_matrix.shape}"
        combined = matrix_data + param_str.encode()
        return hashlib.md5(combined).hexdigest()[:16]

    @performance_tracker("calculate_beta_diversity")
    def calculate_beta_diversity(
        self, otu_matrix: pl.DataFrame, taxonomic_rank: str, metric: str = "braycurtis"
    ) -> DistanceMatrix:
        """
        Calculate beta diversity distance matrix.

        Args:
            otu_matrix: OTU matrix DataFrame
            taxonomic_rank: Taxonomic rank column name
            metric: Distance metric to use

        Returns:
            Beta diversity distance matrix
        """
        try:
            # Validate metric
            if metric not in self.config.analysis.supported_metrics:
                raise AnalysisError(f"Unsupported metric: {metric}")

            self.logger.info(f"Calculating beta diversity using metric: {metric}")

            # Get sample columns and data
            sample_columns = [
                col for col in otu_matrix.columns if col != taxonomic_rank
            ]
            sample_columns.sort()  # Ensure consistent ordering

            # Extract numeric data matrix (samples as rows, taxa as columns)
            # Use optimized approach for large matrices in fast mode
            if hasattr(self.config, "analysis") and self.config.analysis.fast_mode:
                # More efficient extraction for large datasets with memory optimization
                otu_values = otu_matrix.select(sample_columns).to_numpy().T

                # Use float32 for memory savings if configured
                if self.config.analysis.use_float32:
                    otu_values = otu_values.astype(np.float32)
            else:
                otu_values = otu_matrix.select(sample_columns).to_numpy().T

            # Validate matrix
            if np.isnan(otu_values).any():
                raise AnalysisError("OTU matrix contains NaN values")

            if np.isinf(otu_values).any():
                raise AnalysisError("OTU matrix contains infinite values")

            # Calculate beta diversity
            if metric == "bray" or metric == "braycurtis":
                # Use braycurtis for scientific accuracy
                distance_matrix = diversity.beta_diversity(
                    "braycurtis", otu_values, ids=sample_columns
                )
            else:
                distance_matrix = diversity.beta_diversity(
                    metric, otu_values, ids=sample_columns
                )

            # Validate distance matrix
            validation_results = self.validator.validate_distance_matrix(
                distance_matrix.data, sample_columns
            )
            if not validation_results["valid"]:
                raise AnalysisError(
                    f"Invalid distance matrix: {validation_results['errors']}"
                )

            self.logger.info(
                f"Beta diversity calculated successfully: {distance_matrix.shape}"
            )

            return distance_matrix

        except Exception as e:
            self.logger.error(f"Beta diversity calculation failed: {e}")
            raise AnalysisError(f"Beta diversity calculation failed: {e}")

    @performance_tracker("perform_pcoa")
    def perform_pcoa(self, distance_matrix: DistanceMatrix) -> PCoAResults:
        """
        Perform Principal Coordinate Analysis (PCoA).

        Args:
            distance_matrix: Beta diversity distance matrix

        Returns:
            PCoA results
        """
        try:
            self.logger.info("Performing Principal Coordinate Analysis (PCoA)")

            # Perform PCoA with limited dimensions for performance
            # Only compute first 10 dimensions since we only need first 2-5
            n_samples = distance_matrix.shape[0]

            # Use config-based dimension limiting for performance
            if (
                hasattr(self.config, "analysis")
                and hasattr(self.config.analysis, "max_pcoa_dimensions")
                and self.config.analysis.max_pcoa_dimensions is not None
            ):
                max_dimensions = min(
                    self.config.analysis.max_pcoa_dimensions, n_samples - 1
                )
            else:
                max_dimensions = min(10, n_samples - 1)  # Can't exceed n_samples - 1

            ordination_results = pcoa(
                distance_matrix,
                method=self.config.analysis.pcoa_method,
                number_of_dimensions=max_dimensions,
            )

            # Extract results
            eigenvalues = ordination_results.eigvals
            eigenvectors = ordination_results.samples

            # Calculate variance explained
            total_variance = np.sum(eigenvalues[eigenvalues > 0])
            proportion_explained = eigenvalues / total_variance

            # Get sample scores for first two components
            pc1_scores = eigenvectors["PC1"].values
            pc2_scores = eigenvectors["PC2"].values

            sample_scores = {"PC1": pc1_scores, "PC2": pc2_scores}

            # Add more components if available
            for i in range(3, min(len(eigenvalues) + 1, 6)):  # Up to PC5
                pc_name = f"PC{i}"
                if pc_name in eigenvectors.columns:
                    sample_scores[pc_name] = eigenvectors[pc_name].values

            # Calculate variance explained percentages
            variance_explained = {
                "PC1": float(proportion_explained[0] * 100),
                "PC2": float(proportion_explained[1] * 100),
            }

            results = PCoAResults(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                proportion_explained=proportion_explained,
                sample_scores=sample_scores,
                total_variance=total_variance,
            )

            self.logger.info(
                f"PCoA completed: PC1 explains {variance_explained['PC1']:.2f}%, "
                f"PC2 explains {variance_explained['PC2']:.2f}%"
            )

            return results

        except Exception as e:
            self.logger.error(f"PCoA analysis failed: {e}")
            raise AnalysisError(f"PCoA analysis failed: {e}")

    @performance_tracker("perform_permanova")
    def perform_permanova(
        self,
        distance_matrix: DistanceMatrix,
        grouping_data: pl.DataFrame,
        grouping_column: str,
        permutations: int = 999,
    ) -> Optional[Dict[str, Any]]:
        """
        Perform PERMANOVA statistical test.

        Args:
            distance_matrix: Beta diversity distance matrix
            grouping_data: DataFrame with grouping information
            grouping_column: Column name for grouping
            permutations: Number of permutations for test

        Returns:
            PERMANOVA results or None if insufficient variation
        """
        try:
            self.logger.info(f"Performing PERMANOVA with {permutations} permutations")

            # Check if there's enough variation in the grouping variable
            unique_groups = grouping_data.select(grouping_column).unique().height
            if unique_groups < 2:
                self.logger.warning(
                    f"Insufficient variation in grouping variable: only "
                    f"{unique_groups} unique value(s)"
                )
                return None

            # Prepare data for PERMANOVA
            # Filter and align grouping data with distance matrix sample IDs
            valid_samples = list(distance_matrix.ids)

            # Create aligned grouping data
            sample_to_group = dict(
                zip(
                    grouping_data.select("sample_id").to_series(),
                    grouping_data.select(grouping_column).to_series(),
                )
            )

            # Create aligned grouping array
            aligned_grouping = []
            aligned_sample_ids = []

            for sample_id in valid_samples:
                if sample_id in sample_to_group:
                    group_value = sample_to_group[sample_id]
                    if group_value is not None:  # Skip null values
                        aligned_grouping.append(group_value)
                        aligned_sample_ids.append(sample_id)

            if len(aligned_grouping) < self.config.validation.min_samples:
                raise InsufficientDataError(
                    f"Insufficient samples for PERMANOVA: "
                    f"{len(aligned_grouping)} < {self.config.validation.min_samples}"
                )

            # Filter distance matrix to aligned samples
            filtered_dm = distance_matrix.filter(aligned_sample_ids)

            # Perform PERMANOVA
            permanova_results = permanova(
                distance_matrix=filtered_dm,
                grouping=aligned_grouping,
                permutations=permutations,
            )

            # Convert results to serializable format
            results_dict = {}
            for key, value in permanova_results.items():
                if hasattr(value, "tolist") and callable(value.tolist):
                    results_dict[key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    results_dict[key] = value.tolist()
                else:
                    results_dict[key] = value

            self.logger.info(
                f"PERMANOVA completed: "
                f"F-statistic = {results_dict.get('test statistic', 'N/A'):.3f}, "
                f"p-value = {results_dict.get('p-value', 'N/A'):.3f}"
            )

            return results_dict

        except Exception as e:
            self.logger.error(f"PERMANOVA analysis failed: {e}")
            raise AnalysisError(f"PERMANOVA analysis failed: {e}")

    def calculate_dispersion(
        self,
        distance_matrix: DistanceMatrix,
        grouping_data: pl.DataFrame,
        grouping_column: str,
    ) -> Dict[str, Any]:
        """
        Calculate beta dispersion for groups.

        Args:
            distance_matrix: Beta diversity distance matrix
            grouping_data: DataFrame with grouping information
            grouping_column: Column name for grouping

        Returns:
            Beta dispersion results
        """
        try:
            self.logger.info("Calculating beta dispersion")

            # Get unique groups
            groups = grouping_data.select(grouping_column).unique().to_series()

            dispersion_results = {}

            for group in groups:
                if group is None:
                    continue

                # Get samples for this group
                group_samples = (
                    grouping_data.filter(pl.col(grouping_column) == group)
                    .select("sample_id")
                    .to_series()
                    .to_list()
                )

                # Filter samples that exist in distance matrix
                valid_group_samples = [
                    s for s in group_samples if s in distance_matrix.ids
                ]

                if len(valid_group_samples) < 2:
                    continue

                # Calculate distances within group
                group_dm = distance_matrix.filter(valid_group_samples)

                # Calculate dispersion (mean distance to centroid)
                distances = []
                for i, sample1 in enumerate(valid_group_samples):
                    for j, sample2 in enumerate(valid_group_samples):
                        if i < j:  # Only upper triangle
                            dist = group_dm[sample1, sample2]
                            distances.append(dist)

                if distances:
                    dispersion_results[str(group)] = {
                        "mean_distance": np.mean(distances),
                        "std_distance": np.std(distances),
                        "median_distance": np.median(distances),
                        "n_samples": len(valid_group_samples),
                    }

            self.logger.info(
                f"Beta dispersion calculated for {len(dispersion_results)} groups"
            )

            return dispersion_results

        except Exception as e:
            self.logger.error(f"Beta dispersion calculation failed: {e}")
            raise AnalysisError(f"Beta dispersion calculation failed: {e}")

    @performance_tracker("run_complete_analysis")
    def run_complete_analysis(
        self,
        otu_matrix: pl.DataFrame,
        metadata: pl.DataFrame,
        taxonomic_rank: str,
        environmental_param: str,
        metric: str = "braycurtis",
        permutations: int = 999,
    ) -> BetaDiversityResults:
        """
        Run complete beta diversity analysis pipeline.

        Args:
            otu_matrix: OTU matrix DataFrame
            metadata: Metadata DataFrame
            taxonomic_rank: Taxonomic rank column name
            environmental_param: Environmental parameter column
            metric: Distance metric to use
            permutations: Number of permutations for PERMANOVA

        Returns:
            Complete beta diversity results
        """
        try:
            self.logger.info("Starting complete beta diversity analysis")

            # 1. Calculate beta diversity distance matrix
            distance_matrix = self.calculate_beta_diversity(
                otu_matrix, taxonomic_rank, metric
            )

            # 2. Perform PCoA
            pcoa_results = self.perform_pcoa(distance_matrix)

            # 3. Prepare metadata for statistical tests
            sample_columns = [
                col for col in otu_matrix.columns if col != taxonomic_rank
            ]
            aligned_metadata = metadata.filter(
                pl.col("sample_id").is_in(sample_columns)
            ).sort("sample_id")

            # 4. Perform PERMANOVA
            permanova_results = self.perform_permanova(
                distance_matrix, aligned_metadata, environmental_param, permutations
            )

            # 5. Calculate beta dispersion
            dispersion_results = self.calculate_dispersion(
                distance_matrix, aligned_metadata, environmental_param
            )

            # 6. Compile results
            results = BetaDiversityResults(
                distance_matrix=distance_matrix,
                pcoa_results=pcoa_results,
                permanova_results=permanova_results,
                variance_explained={
                    "PC1": pcoa_results.sample_scores["PC1"],
                    "PC2": pcoa_results.sample_scores["PC2"],
                },
                sample_scores=pcoa_results.sample_scores,
                metadata={
                    "metric": metric,
                    "environmental_parameter": environmental_param,
                    "n_samples": len(sample_columns),
                    "n_taxa": otu_matrix.height,
                    "dispersion": dispersion_results,
                    "variance_explained_pc1": float(
                        pcoa_results.proportion_explained[0] * 100
                    ),
                    "variance_explained_pc2": float(
                        pcoa_results.proportion_explained[1] * 100
                    ),
                },
            )

            self.logger.info("Complete beta diversity analysis finished successfully")

            return results

        except Exception as e:
            self.logger.error(f"Complete beta diversity analysis failed: {e}")
            raise AnalysisError(f"Complete beta diversity analysis failed: {e}")

    def calculate_alpha_diversity(
        self, otu_matrix: pl.DataFrame, taxonomic_rank: str, metrics: List[str] = None
    ) -> pl.DataFrame:
        """
        Calculate alpha diversity metrics as complement to beta diversity.

        Args:
            otu_matrix: OTU matrix DataFrame
            taxonomic_rank: Taxonomic rank column name
            metrics: List of alpha diversity metrics to calculate

        Returns:
            DataFrame with alpha diversity results
        """
        try:
            if metrics is None:
                metrics = ["observed_otus", "shannon", "simpson"]

            self.logger.info(f"Calculating alpha diversity metrics: {metrics}")

            sample_columns = [
                col for col in otu_matrix.columns if col != taxonomic_rank
            ]
            otu_values = otu_matrix.select(sample_columns).to_numpy().T

            alpha_results = {"sample_id": sample_columns}

            for metric in metrics:
                if metric == "observed_otus":
                    values = [np.sum(sample > 0) for sample in otu_values]
                elif metric == "shannon":
                    values = [
                        diversity.alpha_diversity("shannon", sample)
                        for sample in otu_values
                    ]
                elif metric == "simpson":
                    values = [
                        diversity.alpha_diversity("simpson", sample)
                        for sample in otu_values
                    ]
                else:
                    try:
                        values = [
                            diversity.alpha_diversity(metric, sample)
                            for sample in otu_values
                        ]
                    except Exception:
                        self.logger.warning(
                            f"Unsupported alpha diversity metric: {metric}"
                        )
                        continue

                alpha_results[metric] = values

            alpha_df = pl.DataFrame(alpha_results)

            self.logger.info(
                f"Alpha diversity calculated for {len(sample_columns)} samples"
            )

            return alpha_df

        except Exception as e:
            self.logger.error(f"Alpha diversity calculation failed: {e}")
            raise AnalysisError(f"Alpha diversity calculation failed: {e}")

    def compare_beta_diversity_metrics(
        self, otu_matrix: pl.DataFrame, taxonomic_rank: str, metrics: List[str] = None
    ) -> Dict[str, DistanceMatrix]:
        """
        Compare multiple beta diversity metrics.

        Args:
            otu_matrix: OTU matrix DataFrame
            taxonomic_rank: Taxonomic rank column name
            metrics: List of metrics to compare

        Returns:
            Dictionary of distance matrices for each metric
        """
        try:
            if metrics is None:
                metrics = ["braycurtis", "jaccard", "euclidean"]

            self.logger.info(f"Comparing beta diversity metrics: {metrics}")

            results = {}

            for metric in metrics:
                if metric in self.config.analysis.supported_metrics:
                    try:
                        dm = self.calculate_beta_diversity(
                            otu_matrix, taxonomic_rank, metric
                        )
                        results[metric] = dm
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate {metric}: {e}")
                else:
                    self.logger.warning(f"Unsupported metric: {metric}")

            self.logger.info(
                f"Successfully calculated {len(results)} beta diversity metrics"
            )

            return results

        except Exception as e:
            self.logger.error(f"Beta diversity comparison failed: {e}")
            raise AnalysisError(f"Beta diversity comparison failed: {e}")

    def calculate_distance_correlation(
        self, dm1: DistanceMatrix, dm2: DistanceMatrix
    ) -> float:
        """
        Calculate Mantel correlation between two distance matrices.

        Args:
            dm1: First distance matrix
            dm2: Second distance matrix

        Returns:
            Correlation coefficient
        """
        try:
            # Ensure both matrices have the same samples
            common_samples = list(set(dm1.ids) & set(dm2.ids))
            common_samples.sort()

            if len(common_samples) < 3:
                raise AnalysisError("Insufficient common samples for correlation")

            # Filter matrices to common samples
            dm1_filtered = dm1.filter(common_samples)
            dm2_filtered = dm2.filter(common_samples)

            # Get distance vectors (upper triangle)
            dist1 = []
            dist2 = []

            for i in range(len(common_samples)):
                for j in range(i + 1, len(common_samples)):
                    dist1.append(dm1_filtered.data[i, j])
                    dist2.append(dm2_filtered.data[i, j])

            # Calculate correlation
            correlation = np.corrcoef(dist1, dist2)[0, 1]

            return float(correlation)

        except Exception as e:
            self.logger.error(f"Distance correlation calculation failed: {e}")
            raise AnalysisError(f"Distance correlation calculation failed: {e}")
