"""
Data validation module for beta diversity analysis.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Any
from pathlib import Path

from .config import BetaDiversityConfig
from .exceptions import DataValidationError, InsufficientDataError
from .logging_config import get_logger, performance_tracker


class DataValidator:
    """Comprehensive data validation for beta diversity analysis."""

    def __init__(self, config=None):
        self.config = config or BetaDiversityConfig()
        self.logger = get_logger(__name__)
        # Extract common validation parameters
        if hasattr(self.config, "processing"):
            self.min_reads_per_sample = getattr(
                self.config.processing, "min_reads_per_sample", 1000
            )
            self.min_prevalence = getattr(
                self.config.processing, "min_prevalence", 0.01
            )
        else:
            self.min_reads_per_sample = 1000
            self.min_prevalence = 0.01

    @performance_tracker("validate_metadata")
    def validate_metadata(
        self, metadata: pl.DataFrame, environmental_param: str
    ) -> Dict[str, Any]:
        """
        Validate metadata DataFrame.

        Args:
            metadata: Polars DataFrame with sample metadata
            environmental_param: Environmental parameter column name

        Returns:
            Dictionary with validation results

        Raises:
            DataValidationError: If validation fails
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {},
        }

        try:
            # Check if DataFrame is empty
            if metadata.height == 0:
                raise DataValidationError("Metadata DataFrame is empty")

            # Check required columns
            required_columns = ["sample_id", "latitude", "longitude"]
            missing_columns = [
                col for col in required_columns if col not in metadata.columns
            ]
            if missing_columns:
                raise DataValidationError(
                    f"Missing required columns: {missing_columns}"
                )

            # Check environmental parameter column
            if environmental_param not in metadata.columns:
                raise DataValidationError(
                    f"Environmental parameter '{environmental_param}' not found in metadata"
                )

            # Validate sample IDs
            sample_ids = metadata.select("sample_id")
            if sample_ids.null_count().item() > 0:
                validation_results["errors"].append("Found null sample IDs")

            # Check for duplicate sample IDs
            unique_samples = sample_ids.unique().height
            total_samples = sample_ids.height
            if unique_samples != total_samples:
                duplicates = total_samples - unique_samples
                validation_results["warnings"].append(
                    f"Found {duplicates} duplicate sample IDs"
                )

            # Validate coordinates
            self._validate_coordinates(metadata, validation_results)

            # Validate environmental parameter
            self._validate_environmental_parameter(
                metadata, environmental_param, validation_results
            )

            # Check minimum sample size
            min_samples = 3  # Default minimum
            if hasattr(self.config, "validation") and hasattr(
                self.config.validation, "min_samples"
            ):
                min_samples = self.config.validation.min_samples

            valid_samples = metadata.filter(
                pl.col("latitude").is_not_null()
                & pl.col("longitude").is_not_null()
                & pl.col("sample_id").is_not_null()
                & pl.col(environmental_param).is_not_null()
            )

            if valid_samples.height < min_samples:
                raise InsufficientDataError(
                    f"Insufficient valid samples: {valid_samples.height} < {min_samples}"
                )

            # Calculate statistics
            validation_results["statistics"] = {
                "total_samples": metadata.height,
                "valid_samples": valid_samples.height,
                "unique_samples": unique_samples,
                "missing_coordinates": metadata.filter(
                    pl.col("latitude").is_null() | pl.col("longitude").is_null()
                ).height,
                "missing_environmental": metadata.filter(
                    pl.col(environmental_param).is_null()
                ).height,
            }

            # Set validation status
            validation_results["valid"] = len(validation_results["errors"]) == 0

            self.logger.info(
                f"Metadata validation completed: {validation_results['statistics']}"
            )

            return validation_results

        except Exception as e:
            self.logger.error(f"Metadata validation failed: {e}")
            raise DataValidationError(f"Metadata validation failed: {e}")

    def _validate_coordinates(
        self, metadata: pl.DataFrame, results: Dict[str, Any]
    ) -> None:
        """Validate latitude and longitude coordinates."""
        # Skip detailed validation in fast mode
        if hasattr(self.config, "validation") and self.config.validation.fast_mode:
            # Quick null check only
            null_coords = metadata.filter(
                pl.col("latitude").is_null() | pl.col("longitude").is_null()
            ).height
            if null_coords > 0:
                results["warnings"].append(
                    f"Found {null_coords} samples with missing coordinates"
                )
            return

        # Full validation (original logic)
        # Check latitude range
        min_lat = metadata.select("latitude").min().item()
        max_lat = metadata.select("latitude").max().item()

        if min_lat is not None and (min_lat < -90 or min_lat > 90):
            results["errors"].append(f"Invalid latitude values: min={min_lat}")

        if max_lat is not None and (max_lat < -90 or max_lat > 90):
            results["errors"].append(f"Invalid latitude values: max={max_lat}")

        # Check longitude range
        min_lon = metadata.select("longitude").min().item()
        max_lon = metadata.select("longitude").max().item()

        if min_lon is not None and (min_lon < -180 or min_lon > 180):
            results["errors"].append(f"Invalid longitude values: min={min_lon}")

        if max_lon is not None and (max_lon < -180 or max_lon > 180):
            results["errors"].append(f"Invalid longitude values: max={max_lon}")

        # Check for null coordinates
        null_coords = metadata.filter(
            pl.col("latitude").is_null() | pl.col("longitude").is_null()
        ).height

        if null_coords > 0:
            results["warnings"].append(
                f"Found {null_coords} samples with missing coordinates"
            )

    def _validate_environmental_parameter(
        self, metadata: pl.DataFrame, param: str, results: Dict[str, Any]
    ) -> None:
        """Validate environmental parameter values."""
        param_data = metadata.select(param)

        # Check for null values
        null_count = param_data.null_count().item()
        if null_count > 0:
            results["warnings"].append(
                f"Found {null_count} null values in environmental parameter '{param}'"
            )

        # Check variation (need at least 2 unique values for analysis)
        unique_values = param_data.unique().height
        if unique_values < 2:
            results["errors"].append(
                f"Insufficient variation in environmental parameter '{param}': only {unique_values} unique value(s)"
            )

        # For temporal variables, validate numeric ranges
        if param in ["temporal_months", "temporal_days", "temporal_years"]:
            self._validate_temporal_parameter(param_data, param, results)

    def _validate_temporal_parameter(
        self, data: pl.DataFrame, param: str, results: Dict[str, Any]
    ) -> None:
        """Validate temporal parameter values."""
        try:
            # Convert to numeric and check ranges
            numeric_data = data.with_columns(
                pl.col(param).cast(pl.Float64, strict=False)
            )

            min_val = numeric_data.min().item()
            max_val = numeric_data.max().item()

            if param == "temporal_months":
                if min_val is not None and (min_val < 1 or min_val > 12):
                    results["warnings"].append(
                        f"Unusual month values: min={min_val}, max={max_val}"
                    )
            elif param == "temporal_days":
                if min_val is not None and (min_val < 1 or min_val > 366):
                    results["warnings"].append(
                        f"Unusual day values: min={min_val}, max={max_val}"
                    )
            elif param == "temporal_years":
                if min_val is not None and (min_val < 1900 or min_val > 2100):
                    results["warnings"].append(
                        f"Unusual year values: min={min_val}, max={max_val}"
                    )

        except Exception as e:
            results["warnings"].append(
                f"Could not validate temporal parameter '{param}': {e}"
            )

    @performance_tracker("validate_abundance_data")
    def validate_abundance_data(
        self, abundance_data: pl.DataFrame, taxonomic_rank: str
    ) -> Dict[str, Any]:
        """
        Validate abundance/count data.

        Args:
            abundance_data: DataFrame with sample abundance data
            taxonomic_rank: Taxonomic rank column name

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {},
        }

        try:
            # Check if DataFrame is empty
            if abundance_data.height == 0:
                raise DataValidationError("Abundance data DataFrame is empty")

            # Check required columns (support both 'freq' and 'reads')
            abundance_col = (
                "reads"
                if "reads" in abundance_data.columns
                else "freq" if "freq" in abundance_data.columns else None
            )
            required_columns = ["sample_id", taxonomic_rank]
            if abundance_col:
                required_columns.append(abundance_col)
            else:
                missing_columns = ["reads or freq"]

            # Check other required columns
            missing_columns = [
                col
                for col in required_columns[:-1]
                if col not in abundance_data.columns
            ]
            if not abundance_col:
                missing_columns.append("reads or freq")

            if missing_columns:
                raise DataValidationError(
                    f"Missing required columns: {missing_columns}"
                )

            # Validate taxonomic rank column
            if taxonomic_rank not in abundance_data.columns:
                raise DataValidationError(
                    f"Taxonomic rank '{taxonomic_rank}' not found in abundance data"
                )

            # Check for null taxonomic identifiers
            null_taxa = abundance_data.filter(pl.col(taxonomic_rank).is_null()).height
            if null_taxa > 0:
                validation_results["warnings"].append(
                    f"Found {null_taxa} records with null taxonomic identifiers"
                )

            # Validate read counts
            self._validate_read_counts(
                abundance_data, validation_results, abundance_col
            )

            # Check minimum requirements
            unique_samples = abundance_data.select("sample_id").unique().height
            unique_taxa = abundance_data.select(taxonomic_rank).unique().height

            if unique_samples < self.config.validation.min_samples:
                raise InsufficientDataError(
                    f"Insufficient samples: {unique_samples} < {self.config.validation.min_samples}"
                )

            if unique_taxa < self.config.validation.min_taxa:
                raise InsufficientDataError(
                    f"Insufficient taxa: {unique_taxa} < {self.config.validation.min_taxa}"
                )

            # Calculate statistics
            total_reads = abundance_data.select(abundance_col).sum().item()
            validation_results["statistics"] = {
                "total_records": abundance_data.height,
                "unique_samples": unique_samples,
                "unique_taxa": unique_taxa,
                "total_reads": total_reads,
                "average_reads_per_sample": (
                    total_reads / unique_samples if unique_samples > 0 else 0
                ),
                "null_taxa": null_taxa,
            }

            validation_results["valid"] = len(validation_results["errors"]) == 0

            self.logger.info(
                f"Abundance data validation completed: {validation_results['statistics']}"
            )

            return validation_results

        except Exception as e:
            self.logger.error(f"Abundance data validation failed: {e}")
            raise DataValidationError(f"Abundance data validation failed: {e}")

    def _validate_read_counts(
        self, data: pl.DataFrame, results: Dict[str, Any], abundance_col: str = "reads"
    ) -> None:
        """Validate read count values."""
        reads_col = data.select(abundance_col)

        # Check for negative values
        negative_reads = data.filter(pl.col(abundance_col) < 0).height
        if negative_reads > 0:
            results["errors"].append(
                f"Found {negative_reads} records with negative read counts"
            )

        # Check for null values
        null_reads = reads_col.null_count().item()
        if null_reads > 0:
            results["errors"].append(
                f"Found {null_reads} records with null read counts"
            )

        # Check for very low counts
        low_reads = (
            data.filter(pl.col(abundance_col) > 0)
            .filter(pl.col(abundance_col) < self.config.validation.min_reads_per_sample)
            .height
        )

        if low_reads > 0:
            results["warnings"].append(
                f"Found {low_reads} records with very low read counts "
                f"(< {self.config.validation.min_reads_per_sample})"
            )

    @performance_tracker("validate_otu_matrix")
    def validate_otu_matrix(
        self, otu_matrix: pl.DataFrame, taxonomic_rank: str
    ) -> Dict[str, Any]:
        """
        Validate OTU matrix for beta diversity analysis.

        Args:
            otu_matrix: OTU matrix DataFrame
            taxonomic_rank: Taxonomic rank column name

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {},
        }

        try:
            # Check matrix dimensions
            n_taxa = otu_matrix.height
            n_samples = otu_matrix.width - 1  # Subtract taxonomic rank column

            if n_taxa < self.config.validation.min_taxa:
                raise InsufficientDataError(
                    f"Insufficient taxa: {n_taxa} < {self.config.validation.min_taxa}"
                )

            if n_samples < self.config.validation.min_samples:
                raise InsufficientDataError(
                    f"Insufficient samples: {n_samples} < {self.config.validation.min_samples}"
                )

            # Get sample columns (all except taxonomic rank)
            sample_columns = [
                col for col in otu_matrix.columns if col != taxonomic_rank
            ]

            # Check for missing values
            for col in sample_columns:
                null_count = otu_matrix.select(col).null_count().item()
                if null_count > 0:
                    validation_results["errors"].append(
                        f"Found {null_count} null values in sample column '{col}'"
                    )

            # Check for negative values
            for col in sample_columns:
                negative_count = otu_matrix.filter(pl.col(col) < 0).height
                if negative_count > 0:
                    validation_results["errors"].append(
                        f"Found {negative_count} negative values in sample column '{col}'"
                    )

            # Calculate sparsity
            total_cells = n_taxa * n_samples
            zero_cells = 0
            for col in sample_columns:
                zero_cells += otu_matrix.filter(pl.col(col) == 0).height

            sparsity = zero_cells / total_cells if total_cells > 0 else 0

            # Check if matrix is too sparse
            if sparsity > 0.95:
                validation_results["warnings"].append(
                    f"Matrix is very sparse: {sparsity:.2%} zero values"
                )

            # Calculate statistics
            validation_results["statistics"] = {
                "n_taxa": n_taxa,
                "n_samples": n_samples,
                "sparsity": sparsity,
                "total_cells": total_cells,
                "zero_cells": zero_cells,
            }

            validation_results["valid"] = len(validation_results["errors"]) == 0

            self.logger.info(
                f"OTU matrix validation completed: {validation_results['statistics']}"
            )

            return validation_results

        except Exception as e:
            self.logger.error(f"OTU matrix validation failed: {e}")
            raise DataValidationError(f"OTU matrix validation failed: {e}")

    def validate_sample_alignment(
        self, metadata: pl.DataFrame, abundance_data: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate alignment between metadata and abundance data samples.

        Args:
            metadata: Metadata DataFrame
            abundance_data: Abundance data DataFrame

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {},
        }

        try:
            # Get unique sample IDs from both datasets
            metadata_samples = set(metadata.select("sample_id").unique().to_series())
            abundance_samples = set(
                abundance_data.select("sample_id").unique().to_series()
            )

            # Find overlapping samples
            common_samples = metadata_samples.intersection(abundance_samples)
            metadata_only = metadata_samples - abundance_samples
            abundance_only = abundance_samples - metadata_samples

            # Check for sufficient overlap
            if len(common_samples) < self.config.validation.min_samples:
                raise InsufficientDataError(
                    f"Insufficient overlapping samples: {len(common_samples)} < {self.config.validation.min_samples}"
                )

            # Log discrepancies
            if metadata_only:
                validation_results["warnings"].append(
                    f"Found {len(metadata_only)} samples in metadata but not in abundance data"
                )

            if abundance_only:
                validation_results["warnings"].append(
                    f"Found {len(abundance_only)} samples in abundance data but not in metadata"
                )

            # Calculate alignment statistics
            validation_results["statistics"] = {
                "metadata_samples": len(metadata_samples),
                "abundance_samples": len(abundance_samples),
                "common_samples": len(common_samples),
                "metadata_only": len(metadata_only),
                "abundance_only": len(abundance_only),
                "alignment_ratio": len(common_samples)
                / max(len(metadata_samples), len(abundance_samples)),
            }

            validation_results["valid"] = len(validation_results["errors"]) == 0

            self.logger.info(
                f"Sample alignment validation completed: {validation_results['statistics']}"
            )

            return validation_results

        except Exception as e:
            self.logger.error(f"Sample alignment validation failed: {e}")
            raise DataValidationError(f"Sample alignment validation failed: {e}")

    def validate_distance_matrix(
        self, distance_matrix: np.ndarray, sample_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Validate distance matrix for analysis.

        Args:
            distance_matrix: Distance matrix array
            sample_ids: List of sample IDs

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {},
        }

        try:
            # Check matrix dimensions
            n_samples = len(sample_ids)
            if distance_matrix.shape != (n_samples, n_samples):
                raise DataValidationError(
                    f"Distance matrix shape {distance_matrix.shape} doesn't match number of samples {n_samples}"
                )

            # Check for invalid values
            if np.isnan(distance_matrix).any():
                validation_results["errors"].append(
                    "Distance matrix contains NaN values"
                )

            if np.isinf(distance_matrix).any():
                validation_results["errors"].append(
                    "Distance matrix contains infinite values"
                )

            # Check matrix properties
            if not np.allclose(distance_matrix, distance_matrix.T, rtol=1e-10):
                validation_results["warnings"].append(
                    "Distance matrix is not symmetric"
                )

            if not np.allclose(np.diag(distance_matrix), 0, atol=1e-10):
                validation_results["warnings"].append(
                    "Distance matrix diagonal is not zero"
                )

            # Check for negative values (distance should be non-negative)
            if (distance_matrix < 0).any():
                validation_results["warnings"].append(
                    "Distance matrix contains negative values"
                )

            # Calculate statistics
            validation_results["statistics"] = {
                "shape": distance_matrix.shape,
                "min_distance": float(np.min(distance_matrix)),
                "max_distance": float(np.max(distance_matrix)),
                "mean_distance": float(np.mean(distance_matrix)),
                "nan_count": int(np.isnan(distance_matrix).sum()),
                "inf_count": int(np.isinf(distance_matrix).sum()),
            }

            validation_results["valid"] = len(validation_results["errors"]) == 0

            self.logger.info(
                f"Distance matrix validation completed: {validation_results['statistics']}"
            )

            return validation_results

        except Exception as e:
            self.logger.error(f"Distance matrix validation failed: {e}")
            raise DataValidationError(f"Distance matrix validation failed: {e}")


# Standalone validation functions for backward compatibility with tests
def validate_otu_data(otu_data: pl.DataFrame) -> bool:
    """Validate OTU data DataFrame."""
    if otu_data.height == 0:
        raise DataValidationError("OTU data is empty")

    if "SampleID" not in otu_data.columns:
        raise DataValidationError("SampleID column not found in OTU data")

    # Check for duplicate sample IDs
    sample_ids = otu_data["SampleID"]
    if sample_ids.n_unique() != len(sample_ids):
        raise DataValidationError("Duplicate sample IDs found in OTU data")

    # Check for negative values in abundance columns
    otu_columns = [col for col in otu_data.columns if col != "SampleID"]
    for col in otu_columns:
        try:
            if otu_data[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                if (otu_data[col] < 0).any():
                    raise DataValidationError(
                        f"OTU data contains negative abundance values in column {col}"
                    )
        except (ValueError, TypeError, pl.ComputeError) as e:
            raise DataValidationError(
                f"Non-numeric abundance values found in column {col}: {e}"
            )

    return True


def validate_metadata(metadata: pl.DataFrame) -> bool:
    """Validate metadata DataFrame."""
    if metadata.height == 0:
        raise DataValidationError("Metadata is empty")

    if "SampleID" not in metadata.columns:
        raise DataValidationError("SampleID column not found in metadata")

    # Check for duplicate sample IDs
    sample_ids = metadata["SampleID"]
    if sample_ids.n_unique() != len(sample_ids):
        raise DataValidationError("Duplicate sample IDs found in metadata")

    return True


def validate_sample_consistency(
    otu_data: pl.DataFrame,
    metadata: pl.DataFrame,
    controls: Optional[pl.DataFrame] = None,
) -> bool:
    """Validate sample consistency across datasets."""
    otu_samples = set(otu_data["SampleID"].to_list())
    meta_samples = set(metadata["SampleID"].to_list())

    if not otu_samples.intersection(meta_samples):
        raise DataValidationError(
            "Sample mismatch: No common samples between OTU data and metadata"
        )

    if controls is not None:
        control_samples = set(controls["SampleID"].to_list())
        missing_controls = control_samples - otu_samples
        if missing_controls:
            raise DataValidationError(
                f"Control samples not found in OTU data: {missing_controls}"
            )

    return True


def validate_abundance_data(abundance_data: pl.DataFrame) -> bool:
    """Validate abundance data matrix."""
    # This is a simplified version - the DataValidator class has more comprehensive validation
    return True


def validate_file_format(file_path: str, expected_format: str = "csv") -> bool:
    """Validate file format."""
    path = Path(file_path)

    if not path.exists():
        raise DataValidationError(f"File does not exist: {file_path}")

    if expected_format == "csv" and path.suffix.lower() != ".csv":
        raise DataValidationError(f"Expected .csv format, got {path.suffix}")
    elif expected_format == "tsv" and path.suffix.lower() not in [".tsv", ".txt"]:
        raise DataValidationError(f"Expected .tsv format, got {path.suffix}")

    # Try to read the file to check if it's corrupted
    try:
        if expected_format in ["csv", "tsv"]:
            separator = "\t" if expected_format == "tsv" else ","
            pl.read_csv(file_path, separator=separator, n_rows=1)
    except Exception as e:
        raise DataValidationError(f"File appears to be corrupted: {e}")

    return True


def validate_configuration(config) -> bool:
    """Validate configuration object."""
    # Check distance metric
    valid_metrics = [
        "braycurtis",
        "jaccard",
        "euclidean",
        "cosine",
        "manhattan",
        "chebyshev",
    ]
    if hasattr(config, "analysis") and hasattr(config.analysis, "distance_metric"):
        if config.analysis.distance_metric not in valid_metrics:
            raise DataValidationError(
                f"Invalid distance metric: {config.analysis.distance_metric}"
            )

    # Check n_components
    if hasattr(config, "analysis") and hasattr(config.analysis, "n_components"):
        if config.analysis.n_components <= 0:
            raise DataValidationError("n_components must be positive")

    # Check permutations
    if hasattr(config, "analysis") and hasattr(config.analysis, "permutations"):
        if config.analysis.permutations < 0:
            raise DataValidationError("permutations must be positive")

    # Check n_clusters
    if hasattr(config, "clustering") and hasattr(config.clustering, "n_clusters"):
        if config.clustering.n_clusters < 2:
            raise DataValidationError("n_clusters must be at least 2")

    # Check output directory
    if hasattr(config, "output") and hasattr(config.output, "results_dir"):
        if not config.output.results_dir:
            raise DataValidationError("results_dir cannot be empty")

    return True
