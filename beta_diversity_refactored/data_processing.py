"""
Data processing module for beta diversity analysis.
"""

import polars as pl
from typing import List, Optional, Tuple
from pathlib import Path

from .config import get_config
from .exceptions import ProcessingError
from .logging_config import get_logger, performance_tracker
from .validation import DataValidator


class DataProcessor:
    """Comprehensive data processing for beta diversity analysis."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        self.validator = DataValidator(config)

    @performance_tracker("load_and_process_metadata")
    def load_and_process_metadata(
        self,
        metadata_path: Optional[Path] = None,
        metadata_df: Optional[pl.DataFrame] = None,
        environmental_param: str = "site",
    ) -> pl.DataFrame:
        """
        Load and process metadata.

        Args:
            metadata_path: Path to metadata file
            metadata_df: Pre-loaded metadata DataFrame
            environmental_param: Environmental parameter column name

        Returns:
            Processed metadata DataFrame
        """
        try:
            # Load metadata if not provided
            if metadata_df is None:
                if metadata_path is None:
                    raise ProcessingError(
                        "Either metadata_path or metadata_df must be provided"
                    )
                metadata_df = self._load_dataframe(metadata_path)

            self.logger.info(f"Processing metadata with {metadata_df.height} records")

            # Validate metadata
            validation_results = self.validator.validate_metadata(
                metadata_df, environmental_param
            )
            if not validation_results["valid"]:
                raise ProcessingError(
                    f"Metadata validation failed: {validation_results['errors']}"
                )

            # Process metadata
            processed_metadata = self._process_metadata_columns(
                metadata_df, environmental_param
            )

            # Filter invalid records
            processed_metadata = self._filter_invalid_metadata(
                processed_metadata, environmental_param
            )

            self.logger.info(
                f"Processed metadata: {processed_metadata.height} valid records"
            )

            return processed_metadata

        except Exception as e:
            self.logger.error(f"Metadata processing failed: {e}")
            raise ProcessingError(f"Metadata processing failed: {e}")

    def _process_metadata_columns(
        self, metadata: pl.DataFrame, environmental_param: str
    ) -> pl.DataFrame:
        """Process and standardize metadata columns."""
        processed = metadata.clone()

        # Ensure sample_id is string type
        if "sample_id" in processed.columns:
            processed = processed.with_columns(pl.col("sample_id").cast(pl.Utf8))

        # Process coordinates with proper precision
        if "latitude" in processed.columns:
            processed = processed.with_columns(
                pl.col("latitude").round(self.config.validation.coordinate_precision)
            )

        if "longitude" in processed.columns:
            processed = processed.with_columns(
                pl.col("longitude").round(self.config.validation.coordinate_precision)
            )

        # Handle special environmental parameters
        processed = self._handle_special_environmental_params(
            processed, environmental_param
        )

        return processed

    def _handle_special_environmental_params(
        self, metadata: pl.DataFrame, env_param: str
    ) -> pl.DataFrame:
        """Handle special environmental parameter processing."""
        processed = metadata.clone()

        # Handle IUCN categories
        if env_param == "iucn_cat":
            processed = processed.with_columns(
                pl.col("iucn_cat").fill_null("not reported")
            )

        # Handle temporal parameters
        if env_param in ["temporal_months", "temporal_days", "temporal_years"]:
            processed = processed.with_columns(
                pl.col(env_param).cast(pl.Float64, strict=False)
            )

        return processed

    def _filter_invalid_metadata(
        self, metadata: pl.DataFrame, environmental_param: str
    ) -> pl.DataFrame:
        """Filter out invalid metadata records."""
        return metadata.filter(
            pl.col("latitude").is_not_null()
            & pl.col("longitude").is_not_null()
            & pl.col("sample_id").is_not_null()
            & pl.col(environmental_param).is_not_null()
        ).unique(subset=["sample_id"], keep="first")

    @performance_tracker("load_and_process_abundance_data")
    def load_and_process_abundance_data(
        self,
        abundance_path: Optional[Path] = None,
        abundance_df: Optional[pl.DataFrame] = None,
        taxonomic_rank: str = "species",
        min_reads_per_sample: int = 100,
        min_reads_per_taxon: int = 10,
    ) -> pl.DataFrame:
        """
        Load and process abundance data.

        Args:
            abundance_path: Path to abundance data file
            abundance_df: Pre-loaded abundance DataFrame
            taxonomic_rank: Taxonomic rank for analysis
            min_reads_per_sample: Minimum reads per sample
            min_reads_per_taxon: Minimum reads per taxon

        Returns:
            Processed abundance DataFrame
        """
        try:
            # Load abundance data if not provided
            if abundance_df is None:
                if abundance_path is None:
                    raise ProcessingError(
                        "Either abundance_path or abundance_df must be provided"
                    )
                abundance_df = self._load_dataframe(abundance_path)

            self.logger.info(
                f"Processing abundance data with {abundance_df.height} records"
            )

            # Validate abundance data
            validation_results = self.validator.validate_abundance_data(
                abundance_df, taxonomic_rank
            )
            if not validation_results["valid"]:
                raise ProcessingError(
                    f"Abundance data validation failed: {validation_results['errors']}"
                )

            # Process abundance data
            processed_abundance = self._process_abundance_columns(
                abundance_df, taxonomic_rank
            )

            # Apply filtering
            processed_abundance = self._filter_abundance_data(
                processed_abundance,
                taxonomic_rank,
                min_reads_per_sample,
                min_reads_per_taxon,
            )

            # Handle taxonomic path creation
            if taxonomic_rank == "taxonomic_path":
                processed_abundance = self._create_taxonomic_path(processed_abundance)

            self.logger.info(
                f"Processed abundance data: {processed_abundance.height} records"
            )

            return processed_abundance

        except Exception as e:
            self.logger.error(f"Abundance data processing failed: {e}")
            raise ProcessingError(f"Abundance data processing failed: {e}")

    def _process_abundance_columns(
        self, abundance_df: pl.DataFrame, taxonomic_rank: str
    ) -> pl.DataFrame:
        """Process and standardize abundance data columns."""
        processed = abundance_df.clone()

        # Ensure sample_id is string type
        if "sample_id" in processed.columns:
            processed = processed.with_columns(pl.col("sample_id").cast(pl.Utf8))

        # Ensure reads column is numeric (handle both 'freq' and 'reads' columns)
        if "freq" in processed.columns and "reads" not in processed.columns:
            processed = processed.rename({"freq": "reads"})

        if "reads" in processed.columns:
            processed = processed.with_columns(
                pl.col("reads").cast(pl.Float64, strict=False)
            )

        # Ensure taxonomic rank column exists and is string type
        if taxonomic_rank in processed.columns:
            processed = processed.with_columns(
                pl.col(taxonomic_rank).cast(pl.Utf8, strict=False)
            )

        return processed

    def _filter_abundance_data(
        self,
        abundance_df: pl.DataFrame,
        taxonomic_rank: str,
        min_reads_per_sample: int,
        min_reads_per_taxon: int,
    ) -> pl.DataFrame:
        """Filter abundance data based on read count thresholds with optimized performance."""
        # Fast mode: minimal filtering for performance
        if hasattr(self.config, "analysis") and self.config.analysis.fast_mode:
            # Quick filter without complex aggregations
            filtered = abundance_df.filter(
                pl.col(taxonomic_rank).is_not_null()
                & (pl.col("reads") >= min_reads_per_taxon)
            )
            self.logger.info(f"Applied filtering: {filtered.height} records remaining")
            return filtered

        # Standard filtering with full validation
        # Remove null taxonomic identifiers
        filtered = abundance_df.filter(pl.col(taxonomic_rank).is_not_null())

        # Remove records with insufficient reads
        filtered = filtered.filter(pl.col("reads") >= min_reads_per_taxon)

        # Calculate per-sample read totals and filter samples
        sample_totals = filtered.group_by("sample_id").agg(
            pl.col("reads").sum().alias("total_reads")
        )

        valid_samples = sample_totals.filter(
            pl.col("total_reads") >= min_reads_per_sample
        ).select("sample_id")

        # Filter to valid samples only
        filtered = filtered.join(valid_samples, on="sample_id", how="inner")

        self.logger.info(f"Applied filtering: {filtered.height} records remaining")

        return filtered

    def _create_taxonomic_path(self, abundance_df: pl.DataFrame) -> pl.DataFrame:
        """Create taxonomic path from individual taxonomic rank columns."""
        taxonomic_columns = [
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]

        # Filter to columns that exist in the DataFrame
        available_columns = [
            col for col in taxonomic_columns if col in abundance_df.columns
        ]

        if not available_columns:
            raise ProcessingError(
                "No taxonomic rank columns found for creating taxonomic path"
            )

        # Create taxonomic path
        return abundance_df.with_columns(
            pl.concat_str(
                [pl.col(col).fill_null("Unknown") for col in available_columns],
                separator=" > ",
            ).alias("taxonomic_path")
        )

    @performance_tracker("create_otu_matrix")
    def create_otu_matrix(
        self,
        abundance_df: pl.DataFrame,
        taxonomic_rank: str,
        aggregate_function: str = "sum",
    ) -> pl.DataFrame:
        """
        Create OTU matrix from abundance data.

        Args:
            abundance_df: Abundance DataFrame
            taxonomic_rank: Taxonomic rank for matrix rows
            aggregate_function: Function to aggregate reads (sum, mean, max)

        Returns:
            OTU matrix DataFrame
        """
        try:
            self.logger.info(
                f"Creating OTU matrix with taxonomic rank: {taxonomic_rank}"
            )

            # Validate inputs
            if taxonomic_rank not in abundance_df.columns:
                raise ProcessingError(
                    f"Taxonomic rank '{taxonomic_rank}' not found in abundance data"
                )

            # Sort for consistent ordering
            sorted_abundance = abundance_df.sort([taxonomic_rank, "sample_id"])

            # Create pivot table (OTU matrix) with optimizations
            if hasattr(self.config, "analysis") and self.config.analysis.fast_mode:
                # For fast mode, use streaming approach for large datasets
                if sorted_abundance.height > 50000:  # Large dataset threshold
                    # Use lazy evaluation for memory efficiency
                    otu_matrix = (
                        sorted_abundance.lazy()
                        .pivot(
                            values="reads",
                            index=taxonomic_rank,
                            on="sample_id",
                            aggregate_function=aggregate_function,
                        )
                        .fill_null(0)
                        .collect()
                    )
                else:
                    otu_matrix = sorted_abundance.pivot(
                        values="reads",
                        index=taxonomic_rank,
                        on="sample_id",
                        aggregate_function=aggregate_function,
                    ).fill_null(0)
            else:
                otu_matrix = sorted_abundance.pivot(
                    values="reads",
                    index=taxonomic_rank,
                    on="sample_id",
                    aggregate_function=aggregate_function,
                ).fill_null(0)

            # Sort matrix by taxonomic rank
            otu_matrix = otu_matrix.sort(taxonomic_rank)

            # Get sample columns and sort them for consistent ordering
            sample_columns = [
                col for col in otu_matrix.columns if col != taxonomic_rank
            ]
            sample_columns.sort()

            # Reorder columns: taxonomic rank first, then sorted sample columns
            otu_matrix = otu_matrix.select([taxonomic_rank] + sample_columns)

            # Ensure all data columns are numeric with memory optimization
            if hasattr(self.config, "analysis") and self.config.analysis.use_float32:
                # Use float32 for memory savings
                otu_matrix = otu_matrix.with_columns(
                    [pl.col(col).cast(pl.Float32) for col in sample_columns]
                )
            else:
                otu_matrix = otu_matrix.with_columns(
                    [pl.col(col).cast(pl.Float64) for col in sample_columns]
                )

            # Validate OTU matrix
            validation_results = self.validator.validate_otu_matrix(
                otu_matrix, taxonomic_rank
            )
            if not validation_results["valid"]:
                raise ProcessingError(
                    f"OTU matrix validation failed: {validation_results['errors']}"
                )

            self.logger.info(
                f"Created OTU matrix: {otu_matrix.height} taxa × {len(sample_columns)} samples"
            )

            return otu_matrix

        except Exception as e:
            self.logger.error(f"OTU matrix creation failed: {e}")
            raise ProcessingError(f"OTU matrix creation failed: {e}")

    @performance_tracker("merge_metadata_abundance")
    def merge_metadata_abundance(
        self,
        metadata: pl.DataFrame,
        abundance_df: pl.DataFrame,
        environmental_param: str,
    ) -> pl.DataFrame:
        """
        Merge metadata with abundance data.

        Args:
            metadata: Metadata DataFrame
            abundance_df: Abundance DataFrame
            environmental_param: Environmental parameter column

        Returns:
            Merged DataFrame
        """
        try:
            # Validate sample alignment
            alignment_results = self.validator.validate_sample_alignment(
                metadata, abundance_df
            )
            if not alignment_results["valid"]:
                raise ProcessingError(
                    f"Sample alignment failed: {alignment_results['errors']}"
                )

            # Select required metadata columns (avoid duplicate column selection)
            metadata_columns = [
                "sample_id",
                environmental_param,
                "latitude",
                "longitude",
            ]
            if environmental_param != "site":
                metadata_columns.append("site")

            metadata_subset = metadata.select(metadata_columns).unique(
                subset=["sample_id"], keep="first"
            )

            # Merge datasets
            merged = abundance_df.join(
                metadata_subset, on="sample_id", how="inner"
            ).drop_nulls(subset=[environmental_param])

            # Sort for consistency
            merged = merged.sort(["sample_id"])

            self.logger.info(
                f"Merged data: {merged.height} records with complete metadata"
            )

            return merged

        except Exception as e:
            self.logger.error(f"Data merging failed: {e}")
            raise ProcessingError(f"Data merging failed: {e}")

    def _load_dataframe(self, file_path: Path) -> pl.DataFrame:
        """Load DataFrame from file with optimized lazy loading."""
        try:
            if not file_path.exists():
                raise ProcessingError(f"File not found: {file_path}")

            # Determine file format and load accordingly with lazy evaluation
            suffix = file_path.suffix.lower()

            if suffix == ".csv":
                # Use lazy loading for large CSV files
                if hasattr(self.config, "analysis") and self.config.analysis.fast_mode:
                    return pl.scan_csv(file_path).collect()
                else:
                    return pl.read_csv(file_path)
            elif suffix in [".tsv", ".txt"]:
                if hasattr(self.config, "analysis") and self.config.analysis.fast_mode:
                    return pl.scan_csv(file_path, separator="\t").collect()
                else:
                    return pl.read_csv(file_path, separator="\t")
            elif suffix == ".parquet":
                return pl.read_parquet(file_path)
            elif suffix == ".json":
                return pl.read_json(file_path)
            else:
                # Try CSV as default
                try:
                    if (
                        hasattr(self.config, "analysis")
                        and self.config.analysis.fast_mode
                    ):
                        return pl.scan_csv(file_path).collect()
                    else:
                        return pl.read_csv(file_path)
                except Exception:
                    # Try TSV as fallback
                    if (
                        hasattr(self.config, "analysis")
                        and self.config.analysis.fast_mode
                    ):
                        return pl.scan_csv(file_path, separator="\t").collect()
                    else:
                        return pl.read_csv(file_path, separator="\t")

        except Exception as e:
            raise ProcessingError(f"Failed to load file {file_path}: {e}")

    def apply_species_filter(
        self, abundance_df: pl.DataFrame, species_list: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Apply species filtering if species list is provided.

        Args:
            abundance_df: Abundance DataFrame
            species_list: Optional list of species to include

        Returns:
            Filtered abundance DataFrame
        """
        if species_list is None or not species_list:
            return abundance_df

        try:
            if "species" not in abundance_df.columns:
                self.logger.warning("Species column not found, skipping species filter")
                return abundance_df

            filtered = abundance_df.filter(pl.col("species").is_in(species_list))

            self.logger.info(
                f"Applied species filter: {filtered.height} records remaining "
                f"(from {abundance_df.height})"
            )

            return filtered

        except Exception as e:
            self.logger.error(f"Species filtering failed: {e}")
            raise ProcessingError(f"Species filtering failed: {e}")

    def apply_sample_filter(
        self, abundance_df: pl.DataFrame, sample_list: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Apply sample filtering if sample list is provided.

        Args:
            abundance_df: Abundance DataFrame
            sample_list: Optional list of sample IDs to include

        Returns:
            Filtered abundance DataFrame
        """
        if sample_list is None or not sample_list:
            return abundance_df

        try:
            filtered = abundance_df.filter(pl.col("sample_id").is_in(sample_list))

            self.logger.info(
                f"Applied sample filter: {filtered.height} records remaining "
                f"(from {abundance_df.height})"
            )

            return filtered

        except Exception as e:
            self.logger.error(f"Sample filtering failed: {e}")
            raise ProcessingError(f"Sample filtering failed: {e}")

    @performance_tracker("prepare_analysis_data")
    def prepare_analysis_data(
        self,
        metadata: pl.DataFrame,
        abundance_df: pl.DataFrame,
        taxonomic_rank: str,
        environmental_param: str,
        species_list: Optional[List[str]] = None,
        sample_list: Optional[List[str]] = None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Prepare all data for beta diversity analysis.

        Args:
            metadata: Metadata DataFrame
            abundance_df: Abundance DataFrame
            taxonomic_rank: Taxonomic rank for analysis
            environmental_param: Environmental parameter
            species_list: Optional species filter
            sample_list: Optional sample filter

        Returns:
            Tuple of (merged_data, otu_matrix)
        """
        try:
            self.logger.info("Preparing data for beta diversity analysis")

            # Apply filters
            filtered_abundance = self.apply_species_filter(abundance_df, species_list)
            filtered_abundance = self.apply_sample_filter(
                filtered_abundance, sample_list
            )

            # Merge metadata and abundance data
            merged_data = self.merge_metadata_abundance(
                metadata, filtered_abundance, environmental_param
            )

            # Create OTU matrix
            otu_matrix = self.create_otu_matrix(merged_data, taxonomic_rank)

            self.logger.info("Data preparation completed successfully")

            return merged_data, otu_matrix

        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise ProcessingError(f"Data preparation failed: {e}")

    def save_processed_data(
        self, data: pl.DataFrame, output_path: Path, format: str = "csv"
    ) -> None:
        """
        Save processed data to file.

        Args:
            data: DataFrame to save
            output_path: Output file path
            format: Output format (csv, tsv, parquet, json)
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "csv":
                data.write_csv(output_path)
            elif format.lower() == "tsv":
                data.write_csv(output_path, separator="\t")
            elif format.lower() == "parquet":
                data.write_parquet(output_path)
            elif format.lower() == "json":
                data.write_json(output_path)
            else:
                raise ProcessingError(f"Unsupported output format: {format}")

            self.logger.info(f"Saved processed data to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            raise ProcessingError(f"Failed to save data: {e}")
