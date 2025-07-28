"""
Beta Diversity Analysis Pipeline

Main orchestration module that coordinates all components for beta diversity analysis.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from analysis import BetaDiversityAnalyzer
from clustering import ClusteringAnalyzer
from config import Config, get_config
from data_processing import DataProcessor
from database import DatabaseManager, create_database_manager
from exceptions import (
    AnalysisError,
    BetaDiversityError,
    DataValidationError,
    InsufficientDataError,
    StorageError,
    VisualizationError,
)
from logging_config import LoggingContext, PerformanceLogger, get_logger
from storage import StorageManager, create_storage_manager
from visualization import BetaDiversityVisualizer

logger = get_logger(__name__)


class BetaDiversityPipeline:
    """
    Main pipeline for beta diversity analysis.

    Orchestrates data processing, analysis, visualization, and storage operations
    following a clean, maintainable architecture.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the beta diversity pipeline.

        Args:
            config: Optional configuration object (uses default if None)
        """
        self.config = config or get_config()
        self.start_time = time.time()

        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.analyzer = BetaDiversityAnalyzer(self.config)
        self.clustering_analyzer = ClusteringAnalyzer(self.config)
        self.visualizer = BetaDiversityVisualizer(self.config)
        self.db_manager = create_database_manager(
            self.config, use_mock=True
        )  # Use mock for challenge
        self.storage_manager = create_storage_manager(self.config)

        # Performance monitoring
        self.performance_logger = PerformanceLogger(logger.logger)

        logger.info("Beta diversity pipeline initialized successfully")

    def run_analysis(
        self,
        metadata: pl.DataFrame,
        taxonomic_data: pl.DataFrame,
        environmental_parameter: str,
        taxonomic_rank: str = "species",
        beta_diversity_metric: str = "braycurtis",
        report_id: str = "beta-analysis",
        project_id: str = "default-project",
        species_list: Optional[pl.DataFrame] = None,
        continuous_variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete beta diversity analysis pipeline.

        Args:
            metadata: Sample metadata DataFrame
            taxonomic_data: Taxonomic abundance data DataFrame
            environmental_parameter: Environmental parameter for analysis
            taxonomic_rank: Taxonomic rank for analysis
            beta_diversity_metric: Beta diversity metric to use
            report_id: Unique identifier for this analysis
            project_id: Project identifier
            species_list: Optional species filtering list
            continuous_variables: List of continuous variable names

        Returns:
            Dictionary with analysis results and metadata

        Raises:
            BetaDiversityError: If analysis fails at any stage
        """
        with LoggingContext(
            logger,
            operation="run_analysis",
            report_id=report_id,
            environmental_parameter=environmental_parameter,
        ):

            try:
                # Update report status to started
                self._update_report_status(report_id, "STARTED", "Analysis initialized")

                # Step 1: Data validation and processing
                with self.performance_logger.time_operation("data_processing"):
                    processed_data = self._process_data(
                        metadata,
                        taxonomic_data,
                        environmental_parameter,
                        taxonomic_rank,
                        species_list,
                    )

                # Step 2: Beta diversity analysis
                with self.performance_logger.time_operation("beta_diversity_analysis"):
                    analysis_results = self._run_beta_diversity_analysis(
                        processed_data, beta_diversity_metric, environmental_parameter
                    )

                # Step 3: Clustering analysis
                with self.performance_logger.time_operation("clustering_analysis"):
                    clustering_results = self._run_clustering_analysis(
                        analysis_results["ordination_result"]
                    )

                # Step 4: Visualization
                with self.performance_logger.time_operation("visualization"):
                    visualization_results = self._create_visualizations(
                        analysis_results,
                        clustering_results,
                        processed_data,
                        environmental_parameter,
                        continuous_variables,
                    )

                # Step 5: Save results
                with self.performance_logger.time_operation("save_results"):
                    output_files = self._save_results(
                        analysis_results,
                        visualization_results,
                        processed_data,
                        report_id,
                        project_id,
                    )

                # Compile final results
                final_results = self._compile_results(
                    analysis_results,
                    clustering_results,
                    visualization_results,
                    output_files,
                    processed_data,
                )

                # Log performance summary
                self._log_performance_summary()

                # Update report status to completed
                self._update_report_status(
                    report_id, "COMPLETED", "Analysis completed successfully"
                )

                logger.info(
                    "Beta diversity analysis completed successfully",
                    report_id=report_id,
                    total_time=time.time() - self.start_time,
                )

                return final_results

            except Exception as e:
                error_msg = f"Beta diversity analysis failed: {str(e)}"
                logger.error(error_msg, error=e)
                self._update_report_status(report_id, "FAILED", error_msg, "error")

                if not isinstance(e, BetaDiversityError):
                    raise BetaDiversityError(error_msg) from e
                raise

    def _process_data(
        self,
        metadata: pl.DataFrame,
        taxonomic_data: pl.DataFrame,
        environmental_parameter: str,
        taxonomic_rank: str,
        species_list: Optional[pl.DataFrame],
    ) -> Dict[str, Any]:
        """Process and validate input data."""
        logger.info("Starting data processing phase")

        # Load and validate data
        validated_metadata = self.data_processor.load_and_validate_metadata(metadata)
        validated_taxonomic = self.data_processor.load_and_validate_taxonomic_data(
            taxonomic_data
        )

        # Filter and merge data
        filtered_metadata, filtered_taxonomic = (
            self.data_processor.filter_and_merge_data(
                validated_metadata,
                validated_taxonomic,
                environmental_parameter,
                taxonomic_rank,
                species_list,
            )
        )

        # Create OTU matrix
        otu_matrix = self.data_processor.create_otu_matrix(
            filtered_taxonomic,
            taxonomic_rank,
            streaming=len(filtered_taxonomic) > 100000,
        )

        # Prepare analysis data
        merged_metadata, sample_ids, otu_array = (
            self.data_processor.prepare_analysis_data(
                filtered_metadata, otu_matrix, environmental_parameter
            )
        )

        logger.info(
            "Data processing completed successfully",
            final_samples=len(sample_ids),
            final_taxa=otu_array.shape[1] if otu_array.size > 0 else 0,
        )

        return {
            "metadata": merged_metadata,
            "sample_ids": sample_ids,
            "otu_array": otu_array,
            "otu_matrix": otu_matrix,
            "taxonomic_rank": taxonomic_rank,
        }

    def _run_beta_diversity_analysis(
        self, processed_data: Dict[str, Any], beta_diversity_metric: str, environmental_parameter: str
    ) -> Dict[str, Any]:
        """Run beta diversity and ordination analysis."""
        logger.info("Starting beta diversity analysis phase")

        # Run complete analysis
        beta_result, ordination_result, permanova_result = (
            self.analyzer.run_complete_analysis(
                processed_data["otu_array"],
                processed_data["sample_ids"],
                processed_data["metadata"],
                environmental_parameter,  # Use the correct environmental parameter
                beta_diversity_metric,
            )
        )

        logger.info("Beta diversity analysis completed successfully")

        return {
            "beta_result": beta_result,
            "ordination_result": ordination_result,
            "permanova_result": permanova_result,
        }

    def _run_clustering_analysis(self, ordination_result) -> Optional[Dict[str, Any]]:
        """Run clustering analysis on ordination results."""
        logger.info("Starting clustering analysis phase")

        try:
            # Extract coordinates
            coordinates = ordination_result.sample_coordinates.select(
                ["PC1", "PC2"]
            ).to_numpy()

            if len(coordinates) < 3:
                logger.warning("Insufficient samples for clustering analysis")
                return None

            # Apply clustering
            clustering_method = self.config.analysis.clustering_method.value
            clustering_result = self.clustering_analyzer.apply_clustering(
                coordinates, method=clustering_method
            )

            logger.info(
                "Clustering analysis completed successfully",
                n_clusters=len(clustering_result.unique_clusters),
            )

            return {"clustering_result": clustering_result, "coordinates": coordinates}

        except Exception as e:
            logger.warning(f"Clustering analysis failed: {e}")
            return None

    def _create_visualizations(
        self,
        analysis_results: Dict[str, Any],
        clustering_results: Optional[Dict[str, Any]],
        processed_data: Dict[str, Any],
        environmental_parameter: str,
        continuous_variables: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Create visualizations for analysis results."""
        logger.info("Starting visualization phase")

        # Create PCoA plot
        fig = self.visualizer.create_pcoa_plot(
            analysis_results["ordination_result"],
            processed_data["metadata"],
            environmental_parameter,
            analysis_results["permanova_result"],
            clustering_results["clustering_result"] if clustering_results else None,
            continuous_variables,
        )

        # Create legend metadata
        is_continuous = (
            continuous_variables is not None
            and environmental_parameter in continuous_variables
        )
        legend_metadata = self.visualizer.create_legend_metadata(
            fig, environmental_parameter, is_continuous
        )

        logger.info("Visualization created successfully")

        return {"figure": fig, "legend_metadata": legend_metadata}

    def _save_results(
        self,
        analysis_results: Dict[str, Any],
        visualization_results: Dict[str, Any],
        processed_data: Dict[str, Any],
        report_id: str,
        project_id: str,
    ) -> Dict[str, str]:
        """Save all analysis results to files."""
        logger.info("Starting results saving phase")

        output_files = {}

        # Save main data output (TSV format like original)
        output_data = self._prepare_output_dataframe(analysis_results, processed_data)

        output_files["tsv_output"] = self.storage_manager.save_dataframe(
            output_data, f"{report_id}_beta_diversity.tsv", format="tsv"
        )

        # Save interactive HTML visualization
        output_files["html_visualization"] = self.visualizer.save_plot(
            visualization_results["figure"],
            f"{self.config.storage.output_dir}/previews/{report_id}_beta_diversity_report.html",
        )

        # Save analysis metadata as JSON
        metadata = self._compile_analysis_metadata(
            analysis_results, visualization_results, processed_data, project_id
        )

        output_files["metadata_json"] = self.storage_manager.save_json(
            metadata, f"{report_id}_metadata.json"
        )

        # Create compressed report JSON (like original)
        report_json = self.visualizer.create_report_json(
            visualization_results["figure"], metadata
        )

        output_files["report_json"] = self.storage_manager.save_json(
            {"datasets": {"results": report_json, "metadata": metadata}},
            f"{report_id}_report.json.gz",
            compress=True,
        )

        # Create tarball with all outputs
        tarball_files = [(f"{report_id}_beta_diversity.tsv", output_data)]

        output_files["tarball"] = self.storage_manager.create_tarball(
            tarball_files, f"{report_id}_results.tar.gz"
        )

        logger.info(
            "Results saving completed successfully", files_created=len(output_files)
        )

        return output_files

    def _prepare_output_dataframe(
        self, analysis_results: Dict[str, Any], processed_data: Dict[str, Any]
    ) -> pl.DataFrame:
        """Prepare final output DataFrame matching original format."""
        # Start with metadata and coordinates
        output_df = processed_data["metadata"].clone()

        # Add PCoA coordinates
        coords_df = analysis_results["ordination_result"].sample_coordinates
        output_df = output_df.join(coords_df, on="sample_id", how="left")

        # Rename PC columns to match original format
        pc_renames = {}
        variance_explained = analysis_results["ordination_result"].variance_explained

        for i in range(1, 3):  # PC1 and PC2
            pc_col = f"PC{i}"
            if pc_col in output_df.columns:
                variance = variance_explained.get(pc_col, 0)
                pc_renames[pc_col] = f"PCoA{i}"

                # Add variance explained columns
                output_df = output_df.with_columns(
                    pl.lit(f"{variance:.2f}%").alias(f"% explained PCoA{i}")
                )

        # Apply renames
        if pc_renames:
            output_df = output_df.rename(pc_renames)

        # Add beta diversity information as string (like original)
        beta_distances = analysis_results["beta_result"].distance_matrix
        sample_ids = analysis_results["beta_result"].sample_ids

        # Convert distance matrix to string representation for each sample
        distance_strings = []
        for i, sample_id in enumerate(sample_ids):
            distances = beta_distances[i, :]
            distance_str = ",".join([f"{d:.6f}" for d in distances])
            distance_strings.append(distance_str)

        # Add distance data
        distance_df = pl.DataFrame(
            {"sample_id": sample_ids, "beta_diversity": distance_strings}
        )

        output_df = output_df.join(distance_df, on="sample_id", how="left")

        # Sort by sample_id for consistent output
        output_df = output_df.sort("sample_id")

        return output_df

    def _compile_analysis_metadata(
        self,
        analysis_results: Dict[str, Any],
        visualization_results: Dict[str, Any],
        processed_data: Dict[str, Any],
        project_id: str,
    ) -> Dict[str, Any]:
        """Compile comprehensive analysis metadata."""
        metadata = {
            "analysis_info": {
                "timestamp": datetime.now().isoformat(),
                "pipeline_version": "1.0.0",
                "project_id": project_id,
                "analysis_type": "beta_diversity",
            },
            "data_info": {
                "total_samples": len(processed_data["sample_ids"]),
                "total_taxa": (
                    processed_data["otu_array"].shape[1]
                    if processed_data["otu_array"].size > 0
                    else 0
                ),
                "taxonomic_rank": processed_data["taxonomic_rank"],
            },
            "analysis_parameters": {
                "beta_diversity_metric": analysis_results["beta_result"].metric,
                "permanova_permutations": analysis_results["permanova_result"].get(
                    "number of permutations", 0
                ),
                "clustering_method": self.config.analysis.clustering_method.value,
            },
            "results_summary": {
                "permanova_f_statistic": analysis_results["permanova_result"].get(
                    "test statistic", 0
                ),
                "permanova_p_value": analysis_results["permanova_result"].get(
                    "p-value", 1
                ),
                "variance_explained_pc1": analysis_results[
                    "ordination_result"
                ].variance_explained.get("PC1", 0),
                "variance_explained_pc2": analysis_results[
                    "ordination_result"
                ].variance_explained.get("PC2", 0),
            },
            "performance_metrics": self.performance_logger.get_timing_summary(),
            "visualization_metadata": visualization_results["legend_metadata"],
        }

        return metadata

    def _compile_results(
        self,
        analysis_results: Dict[str, Any],
        clustering_results: Optional[Dict[str, Any]],
        visualization_results: Dict[str, Any],
        output_files: Dict[str, str],
        processed_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compile final results dictionary."""
        results = {
            "success": True,
            "analysis_results": {
                "beta_diversity": analysis_results["beta_result"],
                "ordination": analysis_results["ordination_result"],
                "permanova": analysis_results["permanova_result"],
            },
            "clustering_results": clustering_results,
            "visualization": {
                "figure": visualization_results["figure"],
                "metadata": visualization_results["legend_metadata"],
            },
            "output_files": output_files,
            "data_summary": {
                "samples_analyzed": len(processed_data["sample_ids"]),
                "taxa_count": (
                    processed_data["otu_array"].shape[1]
                    if processed_data["otu_array"].size > 0
                    else 0
                ),
                "processing_time": time.time() - self.start_time,
            },
            "performance_metrics": self.performance_logger.get_timing_summary(),
        }

        return results

    def _update_report_status(
        self,
        report_id: str,
        status: str,
        message: str,
        error_type: Optional[str] = None,
    ):
        """Update report status in database."""
        try:
            self.db_manager.update_report_status(
                report_id=report_id,
                status=status,
                is_queued=status == "QUEUED",
                error_message=message if status == "FAILED" else None,
                error_type=error_type,
            )
        except Exception as e:
            logger.warning(f"Failed to update report status: {e}")

    def _log_performance_summary(self):
        """Log performance summary."""
        timings = self.performance_logger.get_timing_summary()
        total_time = time.time() - self.start_time

        logger.info(
            "Performance Summary",
            total_execution_time=f"{total_time:.2f}s",
            **{f"{op}_time": f"{time:.2f}s" for op, time in timings.items()},
        )


# Convenience function to run analysis with minimal setup
def run_beta_diversity_analysis(
    metadata_path: str,
    taxonomic_data_path: str,
    environmental_parameter: str,
    config_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run beta diversity analysis from file paths.

    Args:
        metadata_path: Path to metadata CSV/TSV file
        taxonomic_data_path: Path to taxonomic data CSV/TSV file
        environmental_parameter: Environmental parameter for analysis
        config_path: Optional path to configuration file
        **kwargs: Additional arguments for run_analysis

    Returns:
        Analysis results dictionary
    """
    # Load configuration
    from config import ConfigManager

    if config_path:
        config_manager = ConfigManager(config_path)
        config = config_manager.load_config()
    else:
        config = get_config()

    # Initialize pipeline
    pipeline = BetaDiversityPipeline(config)

    # Load data
    metadata = pipeline.storage_manager.load_dataframe(metadata_path)
    taxonomic_data = pipeline.storage_manager.load_dataframe(taxonomic_data_path)

    # Run analysis
    return pipeline.run_analysis(
        metadata=metadata,
        taxonomic_data=taxonomic_data,
        environmental_parameter=environmental_parameter,
        **kwargs,
    )
