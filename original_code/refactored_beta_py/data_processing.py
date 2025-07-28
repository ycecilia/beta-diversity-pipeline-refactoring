"""
Data Processing Module

Handles all data processing operations with Polars optimization and streaming support.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from config import Config
from exceptions import DataValidationError, InsufficientDataError, ProcessingError
from logging_config import LoggingContext, get_logger, log_function_call
from validation import DataValidator, ValidationResult

logger = get_logger(__name__)


class DataProcessor:
    """Optimized data processor using Polars for all DataFrame operations."""

    def __init__(self, config: Config):
        self.config = config
        self.validator = DataValidator(config.validation)

    @log_function_call(logger)
    def load_and_validate_metadata(self, metadata: pl.DataFrame) -> pl.DataFrame:
        """
        Load and validate metadata with comprehensive error checking.

        Args:
            metadata: Raw metadata DataFrame

        Returns:
            Validated and cleaned metadata DataFrame

        Raises:
            DataValidationError: If validation fails
        """
        with LoggingContext(logger, operation="metadata_validation"):
            # Validate metadata
            validation_result = self.validator.validate_metadata(metadata)

            if not validation_result.is_valid:
                error_msg = (
                    f"Metadata validation failed: {'; '.join(validation_result.errors)}"
                )
                logger.error(error_msg)
                raise DataValidationError(error_msg)

            # Log warnings
            for warning in validation_result.warnings:
                logger.warning(f"Metadata validation warning: {warning}")

            # Clean and standardize metadata
            cleaned_metadata = self._clean_metadata(metadata)

            logger.info(
                "Metadata loaded and validated successfully",
                **validation_result.metadata,
            )

            return cleaned_metadata

    def _clean_metadata(self, metadata: pl.DataFrame) -> pl.DataFrame:
        """Clean and standardize metadata."""
        # Remove rows with null sample IDs
        cleaned = metadata.filter(pl.col("sample_id").is_not_null())

        # Remove rows with null coordinates if they exist
        if "latitude" in cleaned.columns and "longitude" in cleaned.columns:
            cleaned = cleaned.filter(
                pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()
            )

        # Remove duplicate sample IDs, keeping first occurrence
        cleaned = cleaned.unique(subset=["sample_id"], maintain_order=True)

        # Sort by sample_id for consistent ordering
        cleaned = cleaned.sort("sample_id")

        return cleaned

    @log_function_call(logger)
    def load_and_validate_taxonomic_data(
        self, taxonomic_data: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Load and validate taxonomic data with comprehensive error checking.

        Args:
            taxonomic_data: Raw taxonomic data DataFrame

        Returns:
            Validated and cleaned taxonomic data DataFrame

        Raises:
            DataValidationError: If validation fails
        """
        with LoggingContext(logger, operation="taxonomic_validation"):
            # Validate taxonomic data
            validation_result = self.validator.validate_taxonomic_data(taxonomic_data)

            if not validation_result.is_valid:
                error_msg = f"Taxonomic data validation failed: {'; '.join(validation_result.errors)}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)

            # Log warnings
            for warning in validation_result.warnings:
                logger.warning(f"Taxonomic data validation warning: {warning}")

            # Clean and standardize taxonomic data
            cleaned_taxonomic = self._clean_taxonomic_data(taxonomic_data)

            logger.info(
                "Taxonomic data loaded and validated successfully",
                **validation_result.metadata,
            )

            return cleaned_taxonomic

    def _clean_taxonomic_data(self, taxonomic_data: pl.DataFrame) -> pl.DataFrame:
        """Clean and standardize taxonomic data."""
        # Remove rows with null sample IDs or frequencies
        cleaned = taxonomic_data.filter(
            pl.col("sample_id").is_not_null() & pl.col("reads").is_not_null()
        )

        # Remove rows with zero or negative frequencies
        cleaned = cleaned.filter(pl.col("reads") > 0)

        # Sort by taxonomic rank and sample_id for consistent ordering
        taxonomic_columns = [
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]
        sort_columns = ["sample_id"]

        # Add available taxonomic columns to sort order
        for col in taxonomic_columns:
            if col in cleaned.columns:
                sort_columns.append(col)
                break  # Sort by the most specific available taxonomic level

        cleaned = cleaned.sort(sort_columns)

        return cleaned

    @log_function_call(logger)
    def filter_and_merge_data(
        self,
        metadata: pl.DataFrame,
        taxonomic_data: pl.DataFrame,
        environmental_parameter: str,
        taxonomic_rank: str,
        species_list: Optional[pl.DataFrame] = None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Filter and merge metadata with taxonomic data.

        Args:
            metadata: Validated metadata DataFrame
            taxonomic_data: Validated taxonomic data DataFrame
            environmental_parameter: Environmental parameter for analysis
            taxonomic_rank: Taxonomic rank for analysis
            species_list: Optional species list for filtering

        Returns:
            Tuple of (filtered_metadata, filtered_taxonomic_data)

        Raises:
            InsufficientDataError: If insufficient data remains after filtering
        """
        with LoggingContext(
            logger,
            operation="filter_and_merge",
            environmental_parameter=environmental_parameter,
            taxonomic_rank=taxonomic_rank,
        ):

            # Validate analysis inputs
            validation_result = self.validator.validate_analysis_inputs(
                metadata, taxonomic_data, environmental_parameter
            )

            if not validation_result.is_valid:
                error_msg = f"Analysis inputs validation failed: {'; '.join(validation_result.errors)}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)

            # Apply species filtering if provided
            if species_list is not None:
                taxonomic_data = self._apply_species_filter(
                    taxonomic_data, species_list
                )

            # Filter by taxonomic rank
            taxonomic_data = self._filter_by_taxonomic_rank(
                taxonomic_data, taxonomic_rank
            )

            # Create taxonomic path if needed
            if taxonomic_rank == "taxonomic_path":
                taxonomic_data = self._create_taxonomic_path(taxonomic_data)

            # Filter metadata for environmental parameter
            filtered_metadata = metadata.filter(
                pl.col(environmental_parameter).is_not_null()
            )

            # Get overlapping samples
            metadata_samples = set(
                filtered_metadata.select("sample_id").to_series().to_list()
            )
            taxonomic_samples = set(
                taxonomic_data.select("sample_id").to_series().to_list()
            )
            valid_samples = list(metadata_samples.intersection(taxonomic_samples))

            if len(valid_samples) < self.config.validation.min_samples_for_analysis:
                raise InsufficientDataError(
                    f"Insufficient samples after filtering: {len(valid_samples)} < {self.config.validation.min_samples_for_analysis}"
                )

            # Filter both datasets to valid samples
            final_metadata = filtered_metadata.filter(
                pl.col("sample_id").is_in(valid_samples)
            ).sort("sample_id")
            final_taxonomic = taxonomic_data.filter(
                pl.col("sample_id").is_in(valid_samples)
            ).sort("sample_id")

            logger.info(
                "Data filtering and merging completed successfully",
                valid_samples=len(valid_samples),
                metadata_samples=len(final_metadata),
                taxonomic_observations=len(final_taxonomic),
            )

            return final_metadata, final_taxonomic

    def _apply_species_filter(
        self, taxonomic_data: pl.DataFrame, species_list: pl.DataFrame
    ) -> pl.DataFrame:
        """Apply species list filtering."""
        if "species" not in taxonomic_data.columns:
            logger.warning("Species column not found, skipping species filtering")
            return taxonomic_data

        species_names = species_list.select(pl.col("name")).unique()
        filtered = taxonomic_data.filter(
            pl.col("species").is_in(species_names.to_series().to_list())
        )

        logger.info(
            f"Applied species filtering",
            original_observations=len(taxonomic_data),
            filtered_observations=len(filtered),
            species_count=len(species_names),
        )

        return filtered

    def _filter_by_taxonomic_rank(
        self, taxonomic_data: pl.DataFrame, taxonomic_rank: str
    ) -> pl.DataFrame:
        """Filter data by taxonomic rank availability."""
        if taxonomic_rank == "taxonomic_path":
            return taxonomic_data  # Will be handled in create_taxonomic_path

        if taxonomic_rank not in taxonomic_data.columns:
            raise ProcessingError(
                f"Taxonomic rank '{taxonomic_rank}' not found in data"
            )

        # Remove taxa that are unknown at the specified rank
        filtered = taxonomic_data.filter(pl.col(taxonomic_rank).is_not_null())

        logger.info(
            f"Filtered by taxonomic rank '{taxonomic_rank}'",
            original_observations=len(taxonomic_data),
            filtered_observations=len(filtered),
        )

        return filtered

    def _create_taxonomic_path(self, taxonomic_data: pl.DataFrame) -> pl.DataFrame:
        """Create taxonomic path column from hierarchy."""
        taxonomic_columns = [
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]
        available_columns = [
            col for col in taxonomic_columns if col in taxonomic_data.columns
        ]

        if not available_columns:
            raise ProcessingError(
                "No taxonomic hierarchy columns found for creating taxonomic path"
            )

        # Create taxonomic path by concatenating available columns
        result = taxonomic_data.with_columns(
            pl.concat_str(
                [pl.col(col).fill_null("Unknown") for col in available_columns],
                separator=" > ",
            ).alias("taxonomic_path")
        )

        logger.info(f"Created taxonomic path using columns: {available_columns}")

        return result

    @log_function_call(logger)
    def create_otu_matrix(
        self, taxonomic_data: pl.DataFrame, taxonomic_rank: str, streaming: bool = False
    ) -> pl.DataFrame:
        """
        Create OTU (Operational Taxonomic Unit) matrix with optimized processing.

        Args:
            taxonomic_data: Filtered taxonomic data
            taxonomic_rank: Taxonomic rank for matrix creation
            streaming: Whether to use streaming processing for large datasets

        Returns:
            OTU matrix with taxa as rows and samples as columns

        Raises:
            ProcessingError: If matrix creation fails
        """
        with LoggingContext(
            logger,
            operation="create_otu_matrix",
            taxonomic_rank=taxonomic_rank,
            streaming=streaming,
        ):

            start_time = time.time()

            try:
                # Select relevant columns
                if taxonomic_rank == "taxonomic_path":
                    matrix_data = taxonomic_data.select(
                        ["sample_id", "reads", "taxonomic_path"]
                    )
                    pivot_index = "taxonomic_path"
                else:
                    matrix_data = taxonomic_data.select(
                        ["sample_id", "reads", taxonomic_rank]
                    )
                    pivot_index = taxonomic_rank

                # Sort data for consistent ordering
                matrix_data = matrix_data.sort([pivot_index, "sample_id"])

                if (
                    streaming and len(matrix_data) > 1000000
                ):  # Use streaming for large datasets
                    otu_matrix = self._create_otu_matrix_streaming(
                        matrix_data, pivot_index
                    )
                else:
                    # Create pivot table (OTU matrix)
                    otu_matrix = matrix_data.pivot(
                        values="reads",
                        index=pivot_index,
                        on="sample_id",
                        aggregate_function="sum",
                    ).fill_null(0)

                # Sort matrix by taxonomic rank for consistent row ordering
                otu_matrix = otu_matrix.sort(pivot_index)

                # Get sample columns and sort them for consistent column ordering
                sample_columns = [
                    col for col in otu_matrix.columns if col != pivot_index
                ]
                sample_columns.sort()

                # Reorder columns: taxonomic rank first, then sorted sample columns
                otu_matrix = otu_matrix.select([pivot_index] + sample_columns)

                # Ensure all data columns are numeric
                otu_matrix = otu_matrix.with_columns(
                    [pl.col(col).cast(pl.Float64) for col in sample_columns]
                )

                processing_time = time.time() - start_time

                logger.info(
                    "OTU matrix created successfully",
                    matrix_shape=otu_matrix.shape,
                    taxa_count=len(otu_matrix),
                    sample_count=len(sample_columns),
                    processing_time=processing_time,
                    streaming_used=streaming and len(matrix_data) > 1000000,
                )

                return otu_matrix

            except Exception as e:
                logger.error("Failed to create OTU matrix", error=e)
                raise ProcessingError(f"OTU matrix creation failed: {str(e)}")

    def _create_otu_matrix_streaming(
        self, matrix_data: pl.DataFrame, pivot_index: str
    ) -> pl.DataFrame:
        """Create OTU matrix using streaming processing for large datasets."""
        logger.info("Using streaming processing for large dataset")

        # For very large datasets, we could implement chunked processing
        # For now, we'll use the standard pivot with lazy evaluation
        return (
            matrix_data.lazy()
            .group_by([pivot_index, "sample_id"])
            .agg(pl.col("reads").sum())
            .collect()
            .pivot(
                values="reads",
                index=pivot_index,
                on="sample_id",
                aggregate_function="sum",
            )
            .fill_null(0)
        )

    @log_function_call(logger)
    def prepare_analysis_data(
        self,
        metadata: pl.DataFrame,
        otu_matrix: pl.DataFrame,
        environmental_parameter: str,
    ) -> Tuple[pl.DataFrame, List[str], np.ndarray]:
        """
        Prepare data for beta diversity analysis.

        Args:
            metadata: Filtered metadata
            otu_matrix: OTU matrix
            environmental_parameter: Environmental parameter for analysis

        Returns:
            Tuple of (merged_metadata, sample_ids, otu_array)

        Raises:
            ProcessingError: If data preparation fails
        """
        with LoggingContext(logger, operation="prepare_analysis_data"):

            try:
                # Get taxonomic rank column (first column is the taxonomic rank)
                taxonomic_rank_col = otu_matrix.columns[0]
                sample_columns = [
                    col for col in otu_matrix.columns if col != taxonomic_rank_col
                ]
                sample_columns.sort()  # Ensure consistent ordering

                # Create sample metadata aligned with OTU matrix
                sample_metadata = pl.DataFrame({"sample_id": sample_columns}).sort(
                    "sample_id"
                )

                # Merge with environmental metadata (preserve all metadata columns)
                merged_metadata = sample_metadata.join(
                    metadata, on="sample_id", how="left"
                ).filter(pl.col(environmental_parameter).is_not_null())

                # Get final valid sample IDs
                valid_sample_ids = (
                    merged_metadata.select("sample_id").to_series().to_list()
                )
                valid_sample_ids.sort()  # Ensure consistent ordering

                # Filter OTU matrix to valid samples and convert to numpy
                valid_columns = [
                    col for col in sample_columns if col in valid_sample_ids
                ]
                otu_array = otu_matrix.select(valid_columns).to_numpy()

                # Transpose for scikit-bio (samples as rows, taxa as columns)
                otu_array = otu_array.T

                logger.info(
                    "Analysis data prepared successfully",
                    final_samples=len(valid_sample_ids),
                    otu_shape=otu_array.shape,
                    taxa_count=otu_array.shape[1],
                )

                return merged_metadata, valid_sample_ids, otu_array

            except Exception as e:
                logger.error("Failed to prepare analysis data", error=e)
                raise ProcessingError(f"Analysis data preparation failed: {str(e)}")

    @log_function_call(logger)
    def optimize_memory_usage(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Optimize memory usage of DataFrame by downcasting numeric types.

        Args:
            df: DataFrame to optimize

        Returns:
            Memory-optimized DataFrame
        """
        optimized = df.clone()

        for col in df.columns:
            dtype = df.schema[col]

            # Optimize integer columns
            if dtype == pl.Int64:
                min_val = df.select(pl.col(col).min()).item()
                max_val = df.select(pl.col(col).max()).item()

                if min_val is not None and max_val is not None:
                    if min_val >= 0 and max_val <= 255:
                        optimized = optimized.with_columns(pl.col(col).cast(pl.UInt8))
                    elif min_val >= -128 and max_val <= 127:
                        optimized = optimized.with_columns(pl.col(col).cast(pl.Int8))
                    elif min_val >= 0 and max_val <= 65535:
                        optimized = optimized.with_columns(pl.col(col).cast(pl.UInt16))
                    elif min_val >= -32768 and max_val <= 32767:
                        optimized = optimized.with_columns(pl.col(col).cast(pl.Int16))
                    elif min_val >= 0 and max_val <= 4294967295:
                        optimized = optimized.with_columns(pl.col(col).cast(pl.UInt32))
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        optimized = optimized.with_columns(pl.col(col).cast(pl.Int32))

            # Optimize float columns
            elif dtype == pl.Float64:
                # Check if values can fit in Float32 without precision loss
                try:
                    float32_test = df.select(
                        pl.col(col).cast(pl.Float32).cast(pl.Float64)
                    )
                    if df.select(pl.col(col)).equals(float32_test):
                        optimized = optimized.with_columns(pl.col(col).cast(pl.Float32))
                except:
                    pass  # Keep as Float64 if conversion fails

        original_memory = df.estimated_size()
        optimized_memory = optimized.estimated_size()
        memory_savings = original_memory - optimized_memory

        if memory_savings > 0:
            logger.info(
                f"Memory optimization completed",
                original_mb=original_memory / 1024 / 1024,
                optimized_mb=optimized_memory / 1024 / 1024,
                savings_mb=memory_savings / 1024 / 1024,
                savings_percent=(memory_savings / original_memory) * 100,
            )

        return optimized

    @log_function_call(logger)
    def cache_dataframe(self, df: pl.DataFrame, cache_key: str) -> str:
        """
        Cache DataFrame to disk for reuse.

        Args:
            df: DataFrame to cache
            cache_key: Unique key for caching

        Returns:
            Path to cached file
        """
        cache_dir = Path(self.config.storage.cache_dir)
        cache_file = cache_dir / f"{cache_key}.parquet"

        # Write to parquet for efficient storage and loading
        df.write_parquet(cache_file)

        logger.info(
            f"DataFrame cached successfully",
            cache_key=cache_key,
            cache_file=str(cache_file),
            df_shape=df.shape,
        )

        return str(cache_file)

    @log_function_call(logger)
    def load_cached_dataframe(self, cache_key: str) -> Optional[pl.DataFrame]:
        """
        Load cached DataFrame from disk.

        Args:
            cache_key: Cache key to load

        Returns:
            Cached DataFrame or None if not found
        """
        cache_dir = Path(self.config.storage.cache_dir)
        cache_file = cache_dir / f"{cache_key}.parquet"

        if cache_file.exists():
            try:
                df = pl.read_parquet(cache_file)
                logger.info(
                    f"DataFrame loaded from cache",
                    cache_key=cache_key,
                    df_shape=df.shape,
                )
                return df
            except Exception as e:
                logger.warning(f"Failed to load cached DataFrame: {e}")

        return None
