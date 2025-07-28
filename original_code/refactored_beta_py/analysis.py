"""
Beta Diversity Analysis Module

Handles beta diversity calculations, ordination, and statistical analysis.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from config import BetaDiversityMetric, Config
from exceptions import AnalysisError, InsufficientDataError
from logging_config import LoggingContext, get_logger, log_function_call
from skbio import diversity
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa
from validation import DataValidator

logger = get_logger(__name__)


@dataclass
class BetaDiversityResult:
    """Results from beta diversity analysis."""

    distance_matrix: np.ndarray
    sample_ids: List[str]
    metric: str
    ordination_results: Optional[Any] = None
    permanova_results: Optional[Dict[str, Any]] = None
    variance_explained: Optional[Dict[str, float]] = None


@dataclass
class OrdinationResult:
    """Results from ordination analysis."""

    sample_coordinates: pl.DataFrame
    eigenvalues: np.ndarray
    variance_explained: Dict[str, float]
    method: str


class BetaDiversityAnalyzer:
    """Handles beta diversity calculations and statistical analysis."""

    def __init__(self, config: Config):
        self.config = config
        self.validator = DataValidator(config.validation)

    @log_function_call(logger)
    def calculate_beta_diversity(
        self, otu_matrix: np.ndarray, sample_ids: List[str], metric: str = "braycurtis"
    ) -> BetaDiversityResult:
        """
        Calculate beta diversity distance matrix.

        Args:
            otu_matrix: OTU matrix (samples x taxa)
            sample_ids: Sample identifiers
            metric: Beta diversity metric to use

        Returns:
            BetaDiversityResult with distance matrix and metadata

        Raises:
            AnalysisError: If beta diversity calculation fails
        """
        with LoggingContext(
            logger,
            operation="calculate_beta_diversity",
            metric=metric,
            sample_count=len(sample_ids),
        ):

            try:
                # Validate inputs
                self._validate_otu_matrix(otu_matrix, sample_ids)

                # Convert metric name if needed
                skbio_metric = self._convert_metric_name(metric)

                logger.info(f"Calculating beta diversity using {skbio_metric} metric")

                # Calculate beta diversity matrix
                distance_matrix = diversity.beta_diversity(
                    skbio_metric, otu_matrix, ids=sample_ids
                )

                # Validate resulting distance matrix
                validation_result = self.validator.validate_distance_matrix(
                    distance_matrix.data, sample_ids
                )

                if not validation_result.is_valid:
                    error_msg = f"Invalid distance matrix: {'; '.join(validation_result.errors)}"
                    logger.error(error_msg)
                    raise AnalysisError(error_msg)

                # Log warnings if any
                for warning in validation_result.warnings:
                    logger.warning(f"Distance matrix validation warning: {warning}")

                logger.info(
                    "Beta diversity calculation completed successfully",
                    **validation_result.metadata,
                )

                return BetaDiversityResult(
                    distance_matrix=distance_matrix.data,
                    sample_ids=list(distance_matrix.ids),
                    metric=metric,
                )

            except Exception as e:
                logger.error("Beta diversity calculation failed", error=e)
                if isinstance(e, AnalysisError):
                    raise
                raise AnalysisError(f"Beta diversity calculation failed: {str(e)}")

    def _validate_otu_matrix(self, otu_matrix: np.ndarray, sample_ids: List[str]):
        """Validate OTU matrix for beta diversity analysis."""
        if otu_matrix.ndim != 2:
            raise AnalysisError(f"OTU matrix must be 2D, got {otu_matrix.ndim}D")

        if otu_matrix.shape[0] != len(sample_ids):
            raise AnalysisError(
                f"OTU matrix rows ({otu_matrix.shape[0]}) must match sample count ({len(sample_ids)})"
            )

        if otu_matrix.shape[0] < self.config.validation.min_samples_for_analysis:
            raise InsufficientDataError(
                f"Insufficient samples for analysis: {otu_matrix.shape[0]} < {self.config.validation.min_samples_for_analysis}"
            )

        # Check for invalid values
        if np.isnan(otu_matrix).any():
            raise AnalysisError("OTU matrix contains NaN values")

        if np.isinf(otu_matrix).any():
            raise AnalysisError("OTU matrix contains infinite values")

        if (otu_matrix < 0).any():
            raise AnalysisError("OTU matrix contains negative values")

    def _convert_metric_name(self, metric: str) -> str:
        """Convert metric name to scikit-bio format."""
        metric_mapping = {
            "bray": "braycurtis",
            "jaccard": "jaccard",
            "braycurtis": "braycurtis",
        }

        converted = metric_mapping.get(metric.lower(), metric.lower())

        # Validate metric is supported by scikit-bio
        try:
            # Test with dummy data to ensure metric is valid
            dummy_data = np.array([[1, 0], [0, 1]])
            diversity.beta_diversity(converted, dummy_data, ids=["A", "B"])
        except Exception as e:
            raise AnalysisError(
                f"Unsupported beta diversity metric '{metric}': {str(e)}"
            )

        return converted

    @log_function_call(logger)
    def perform_ordination(
        self,
        beta_result: BetaDiversityResult,
        method: str = "pcoa",
        dimensions: int = 2,
    ) -> OrdinationResult:
        """
        Perform ordination analysis on distance matrix.

        Args:
            beta_result: Beta diversity results
            method: Ordination method (currently only PCoA supported)
            dimensions: Number of dimensions to extract

        Returns:
            OrdinationResult with sample coordinates and metadata

        Raises:
            AnalysisError: If ordination fails
        """
        with LoggingContext(
            logger, operation="perform_ordination", method=method, dimensions=dimensions
        ):

            try:
                if method.lower() != "pcoa":
                    raise AnalysisError(f"Unsupported ordination method: {method}")

                # Create distance matrix object for scikit-bio
                from skbio import DistanceMatrix

                distance_matrix = DistanceMatrix(
                    beta_result.distance_matrix, ids=beta_result.sample_ids
                )

                # Perform PCoA
                logger.info("Performing Principal Coordinate Analysis (PCoA)")
                ordination_results = pcoa(distance_matrix)

                # Extract sample coordinates
                coordinates_data = {}
                coordinates_data["sample_id"] = list(ordination_results.samples.index)

                for i in range(min(dimensions, len(ordination_results.eigvals))):
                    pc_name = f"PC{i+1}"
                    coordinates_data[pc_name] = list(
                        ordination_results.samples.iloc[:, i]
                    )

                coordinates_df = pl.DataFrame(coordinates_data).sort("sample_id")

                # Calculate variance explained
                total_variance = sum(ordination_results.eigvals)
                variance_explained = {}

                for i in range(min(dimensions, len(ordination_results.eigvals))):
                    pc_name = f"PC{i+1}"
                    variance = ordination_results.eigvals.iloc[i] / total_variance * 100
                    variance_explained[pc_name] = float(variance)

                logger.info(
                    "Ordination analysis completed successfully",
                    eigenvalues_count=len(ordination_results.eigvals),
                    variance_explained_pc1=variance_explained.get("PC1", 0),
                    variance_explained_pc2=variance_explained.get("PC2", 0),
                )

                return OrdinationResult(
                    sample_coordinates=coordinates_df,
                    eigenvalues=ordination_results.eigvals.values,
                    variance_explained=variance_explained,
                    method=method.upper(),
                )

            except Exception as e:
                logger.error("Ordination analysis failed", error=e)
                if isinstance(e, AnalysisError):
                    raise
                raise AnalysisError(f"Ordination analysis failed: {str(e)}")

    @log_function_call(logger)
    def perform_permanova(
        self,
        beta_result: BetaDiversityResult,
        grouping_data: pl.DataFrame,
        grouping_column: str,
        permutations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform PERMANOVA analysis to test for differences between groups.

        Args:
            beta_result: Beta diversity results
            grouping_data: DataFrame with sample grouping information
            grouping_column: Column name for grouping variable
            permutations: Number of permutations (uses config default if None)

        Returns:
            PERMANOVA results dictionary

        Raises:
            AnalysisError: If PERMANOVA fails
        """
        with LoggingContext(
            logger, operation="perform_permanova", grouping_column=grouping_column
        ):

            try:
                if permutations is None:
                    permutations = self.config.analysis.permanova_permutations

                # Validate grouping data
                self._validate_grouping_data(
                    grouping_data, grouping_column, beta_result.sample_ids
                )

                # Create aligned grouping array
                aligned_grouping = self._create_aligned_grouping(
                    beta_result.sample_ids, grouping_data, grouping_column
                )

                # Check for sufficient variation
                unique_groups = len(set(aligned_grouping))
                if unique_groups < 2:
                    raise InsufficientDataError(
                        f"Insufficient group variation for PERMANOVA: only {unique_groups} unique group(s)"
                    )

                # Validate group sizes
                self._validate_group_sizes(aligned_grouping)

                # Create distance matrix object for scikit-bio
                from skbio import DistanceMatrix

                distance_matrix = DistanceMatrix(
                    beta_result.distance_matrix, ids=beta_result.sample_ids
                )

                logger.info(
                    f"Performing PERMANOVA with {permutations} permutations",
                    unique_groups=unique_groups,
                    total_samples=len(aligned_grouping),
                )

                # Perform PERMANOVA
                permanova_results = permanova(
                    distance_matrix=distance_matrix,
                    grouping=aligned_grouping,
                    permutations=permutations,
                )

                # Convert results to serializable format
                results_dict = self._convert_permanova_results(permanova_results)

                logger.info(
                    "PERMANOVA analysis completed successfully",
                    test_statistic=results_dict["test statistic"],
                    p_value=results_dict["p-value"],
                    permutations=results_dict["number of permutations"],
                )

                return results_dict

            except Exception as e:
                logger.error("PERMANOVA analysis failed", error=e)
                if isinstance(e, (AnalysisError, InsufficientDataError)):
                    raise
                raise AnalysisError(f"PERMANOVA analysis failed: {str(e)}")

    def _validate_grouping_data(
        self, grouping_data: pl.DataFrame, grouping_column: str, sample_ids: List[str]
    ):
        """Validate grouping data for PERMANOVA."""
        if grouping_column not in grouping_data.columns:
            raise AnalysisError(
                f"Grouping column '{grouping_column}' not found in data"
            )

        if "sample_id" not in grouping_data.columns:
            raise AnalysisError("Sample ID column not found in grouping data")

        # Check sample overlap
        grouping_samples = set(grouping_data.select("sample_id").to_series().to_list())
        analysis_samples = set(sample_ids)
        overlap = grouping_samples.intersection(analysis_samples)

        if len(overlap) < len(sample_ids):
            missing_samples = len(sample_ids) - len(overlap)
            logger.warning(f"Missing grouping data for {missing_samples} samples")

    def _create_aligned_grouping(
        self, sample_ids: List[str], grouping_data: pl.DataFrame, grouping_column: str
    ) -> List[Any]:
        """Create grouping array aligned with sample IDs."""
        # Create lookup dictionary
        grouping_lookup = dict(
            zip(
                grouping_data.select("sample_id").to_series().to_list(),
                grouping_data.select(grouping_column).to_series().to_list(),
            )
        )

        # Create aligned grouping
        aligned_grouping = []
        for sample_id in sample_ids:
            if sample_id in grouping_lookup:
                aligned_grouping.append(grouping_lookup[sample_id])
            else:
                raise AnalysisError(f"No grouping data found for sample: {sample_id}")

        return aligned_grouping

    def _validate_group_sizes(self, grouping: List[Any]):
        """Validate that groups have sufficient sample sizes."""
        from collections import Counter

        group_counts = Counter(grouping)

        small_groups = [
            group
            for group, count in group_counts.items()
            if count < self.config.validation.min_samples_per_group
        ]

        if small_groups:
            logger.warning(
                f"Groups with small sample sizes: {dict((g, group_counts[g]) for g in small_groups)}"
            )

    def _convert_permanova_results(self, permanova_results) -> Dict[str, Any]:
        """Convert PERMANOVA results to serializable dictionary."""
        results_dict = {}

        for key, value in permanova_results.items():
            if hasattr(value, "tolist") and callable(value.tolist):
                # Convert numpy arrays to lists
                results_dict[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                # Convert numpy scalars to Python types
                results_dict[key] = value.item()
            else:
                results_dict[key] = value

        return results_dict

    @log_function_call(logger)
    def run_complete_analysis(
        self,
        otu_matrix: np.ndarray,
        sample_ids: List[str],
        grouping_data: pl.DataFrame,
        grouping_column: str,
        metric: str = "braycurtis",
    ) -> Tuple[BetaDiversityResult, OrdinationResult, Dict[str, Any]]:
        """
        Run complete beta diversity analysis pipeline.

        Args:
            otu_matrix: OTU matrix (samples x taxa)
            sample_ids: Sample identifiers
            grouping_data: DataFrame with grouping information
            grouping_column: Column for grouping variable
            metric: Beta diversity metric

        Returns:
            Tuple of (beta_diversity_result, ordination_result, permanova_result)
        """
        with LoggingContext(logger, operation="complete_beta_analysis"):

            # Calculate beta diversity
            beta_result = self.calculate_beta_diversity(otu_matrix, sample_ids, metric)

            # Perform ordination
            ordination_result = self.perform_ordination(
                beta_result, dimensions=self.config.analysis.pcoa_dimensions
            )

            # Perform PERMANOVA
            permanova_result = self.perform_permanova(
                beta_result, grouping_data, grouping_column
            )

            logger.info("Complete beta diversity analysis finished successfully")

            return beta_result, ordination_result, permanova_result

    def calculate_distance_matrix(
        self, abundance_matrix: pl.DataFrame, metric: str = "bray_curtis"
    ) -> pl.DataFrame:
        """
        Calculate distance matrix from abundance matrix.

        Args:
            abundance_matrix: Sample x OTU abundance matrix
            metric: Distance metric to use

        Returns:
            Distance matrix in long format (sample_1, sample_2, distance)
        """
        try:
            # Convert to numpy array for distance calculation
            sample_ids = abundance_matrix["sample_id"].to_list()
            otu_columns = [
                col for col in abundance_matrix.columns if col != "sample_id"
            ]

            # Get abundance data as numpy array
            abundance_data = abundance_matrix.select(otu_columns).to_numpy()

            # Calculate beta diversity using scikit-bio
            skbio_metric = self._convert_metric_name(metric)
            distance_matrix = diversity.beta_diversity(
                skbio_metric, abundance_data, ids=sample_ids
            )

            # Convert to long format DataFrame
            distance_data = []
            for i, sample1 in enumerate(sample_ids):
                for j, sample2 in enumerate(sample_ids):
                    distance_data.append(
                        {
                            "sample_1": sample1,
                            "sample_2": sample2,
                            "distance": distance_matrix.data[i, j],
                        }
                    )

            return pl.DataFrame(distance_data)

        except Exception as e:
            logger.error("Distance matrix calculation failed", error=e)
            raise AnalysisError(f"Distance matrix calculation failed: {str(e)}")

    def calculate_dispersion(
        self,
        distance_matrix: pl.DataFrame,
        metadata: pl.DataFrame,
        grouping_column: str = "site",
    ) -> Dict[str, Any]:
        """
        Calculate beta dispersion.

        Args:
            distance_matrix: Distance matrix DataFrame
            metadata: Sample metadata DataFrame
            grouping_column: Column to use for grouping

        Returns:
            Dispersion results dictionary
        """
        try:
            # This is a simplified dispersion calculation
            # In practice, you would use proper multivariate dispersion methods
            samples = sorted(set(distance_matrix["sample_1"].to_list()))
            metadata_filtered = metadata.filter(pl.col("sample_id").is_in(samples))

            dispersion_by_group = {}
            for group in metadata_filtered[grouping_column].unique():
                group_samples = metadata_filtered.filter(
                    pl.col(grouping_column) == group
                )["sample_id"].to_list()

                # Calculate mean distance within group
                group_distances = distance_matrix.filter(
                    (pl.col("sample_1").is_in(group_samples))
                    & (pl.col("sample_2").is_in(group_samples))
                    & (pl.col("sample_1") != pl.col("sample_2"))
                )["distance"].mean()

                dispersion_by_group[group] = group_distances

            return {
                "dispersion_by_group": dispersion_by_group,
                "grouping_variable": grouping_column,
            }

        except Exception as e:
            logger.error("Dispersion calculation failed", error=e)
            raise AnalysisError(f"Dispersion calculation failed: {str(e)}")

    def perform_anosim(
        self,
        distance_matrix: pl.DataFrame,
        metadata: pl.DataFrame,
        grouping_column: str = "site",
    ) -> Dict[str, Any]:
        """
        Perform ANOSIM analysis.

        Args:
            distance_matrix: Distance matrix DataFrame
            metadata: Sample metadata DataFrame
            grouping_column: Column to use for grouping

        Returns:
            ANOSIM results dictionary
        """
        try:
            # Convert distance matrix to scikit-bio format
            samples = sorted(set(distance_matrix["sample_1"].to_list()))
            matrix_data = np.zeros((len(samples), len(samples)))

            for row in distance_matrix.iter_rows(named=True):
                i = samples.index(row["sample_1"])
                j = samples.index(row["sample_2"])
                matrix_data[i, j] = row["distance"]

            distance_matrix_obj = DistanceMatrix(matrix_data, samples)

            # Get grouping data
            metadata_filtered = metadata.filter(pl.col("sample_id").is_in(samples))
            grouping = metadata_filtered.sort("sample_id")[grouping_column].to_list()

            # Run ANOSIM
            results = anosim(distance_matrix_obj, grouping, permutations=999)

            return {
                "test_statistic": float(results["test statistic"]),
                "p_value": float(results["p-value"]),
                "number_of_permutations": int(results["number of permutations"]),
                "grouping_variable": grouping_column,
            }

        except Exception as e:
            logger.error("ANOSIM analysis failed", error=e)
            raise AnalysisError(f"ANOSIM analysis failed: {str(e)}")

    def calculate_alpha_diversity(self, abundance_matrix: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate alpha diversity metrics.

        Args:
            abundance_matrix: Sample x OTU abundance matrix

        Returns:
            DataFrame with alpha diversity metrics per sample
        """
        try:
            alpha_results = []
            sample_ids = abundance_matrix["sample_id"].to_list()
            otu_columns = [
                col for col in abundance_matrix.columns if col != "sample_id"
            ]

            for i, sample_id in enumerate(sample_ids):
                # Get abundance values for this sample
                abundances = abundance_matrix.row(i)[1:]  # Skip sample_id column

                # Calculate metrics
                shannon = diversity.alpha_diversity("shannon", [abundances])[0]
                simpson = diversity.alpha_diversity("simpson", [abundances])[0]
                richness = (np.array(abundances) > 0).sum()

                alpha_results.append(
                    {
                        "sample_id": sample_id,
                        "shannon": float(shannon),
                        "simpson": float(simpson),
                        "richness": int(richness),
                    }
                )

            return pl.DataFrame(alpha_results)

        except Exception as e:
            logger.error("Alpha diversity calculation failed", error=e)
            raise AnalysisError(f"Alpha diversity calculation failed: {str(e)}")

    def perform_mantel_test(
        self, matrix1: pl.DataFrame, matrix2: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        Perform Mantel test between two distance matrices.

        Args:
            matrix1: First distance matrix
            matrix2: Second distance matrix

        Returns:
            Mantel test results
        """
        try:
            # Convert both matrices to scikit-bio format
            samples = sorted(set(matrix1["sample_1"].to_list()))

            data1 = np.zeros((len(samples), len(samples)))
            data2 = np.zeros((len(samples), len(samples)))

            for row in matrix1.iter_rows(named=True):
                i = samples.index(row["sample_1"])
                j = samples.index(row["sample_2"])
                data1[i, j] = row["distance"]

            for row in matrix2.iter_rows(named=True):
                i = samples.index(row["sample_1"])
                j = samples.index(row["sample_2"])
                data2[i, j] = row["distance"]

            dm1 = DistanceMatrix(data1, samples)
            dm2 = DistanceMatrix(data2, samples)

            # Perform Mantel test
            coeff, p_value, n = mantel(dm1, dm2, permutations=999)

            return {
                "correlation_coefficient": float(coeff),
                "p_value": float(p_value),
                "number_of_permutations": int(n),
            }

        except Exception as e:
            logger.error("Mantel test failed", error=e)
            raise AnalysisError(f"Mantel test failed: {str(e)}")

    def calculate_pcoa(self, distance_matrix: pl.DataFrame) -> Dict[str, Any]:
        """
        Perform Principal Coordinates Analysis (PCoA).

        Args:
            distance_matrix: Distance matrix DataFrame

        Returns:
            PCoA results dictionary
        """
        try:
            # Convert distance matrix to scikit-bio format
            samples = sorted(set(distance_matrix["sample_1"].to_list()))
            matrix_data = np.zeros((len(samples), len(samples)))

            for row in distance_matrix.iter_rows(named=True):
                i = samples.index(row["sample_1"])
                j = samples.index(row["sample_2"])
                matrix_data[i, j] = row["distance"]

            distance_matrix_obj = DistanceMatrix(matrix_data, samples)

            # Perform PCoA
            pcoa_results = pcoa(distance_matrix_obj)

            # Create coordinates DataFrame
            coordinates_data = []
            for i, sample_id in enumerate(samples):
                coordinates_data.append(
                    {
                        "sample_id": sample_id,
                        "PC1": float(pcoa_results.samples.iloc[i, 0]),
                        "PC2": float(pcoa_results.samples.iloc[i, 1]),
                    }
                )

            # Calculate explained variance
            total_eigenvals = pcoa_results.eigvals.sum()
            explained_variance = [
                float(pcoa_results.eigvals.iloc[i] / total_eigenvals * 100)
                for i in range(min(3, len(pcoa_results.eigvals)))
            ]

            return {
                "coordinates": pl.DataFrame(coordinates_data),
                "explained_variance": explained_variance,
                "eigenvalues": pcoa_results.eigvals.tolist()[:3],
            }

        except Exception as e:
            logger.error("PCoA calculation failed", error=e)
            raise AnalysisError(f"PCoA calculation failed: {str(e)}")
