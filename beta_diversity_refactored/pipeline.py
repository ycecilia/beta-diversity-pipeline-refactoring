"""
Main beta diversity analysis pipeline.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import polars as pl
import numpy as np
from dataclasses import dataclass

from .config import get_config, BetaDiversityConfig
from .exceptions import (
    AnalysisError,
    DataValidationError,
)
from .logging_config import (
    get_logger,
    performance_tracker,
    PerformanceLogger,
    ProgressTracker,
)
from .validation import DataValidator
from .data_processing import DataProcessor
from .analysis import BetaDiversityAnalyzer, BetaDiversityResults
from .visualization import BetaDiversityVisualizer
from .clustering import ClusteringAnalyzer
from .storage import StorageManager


@dataclass
class PipelineInputs:
    """Input parameters for beta diversity pipeline."""

    metadata_path: Optional[Path] = None
    abundance_path: Optional[Path] = None
    metadata_df: Optional[pl.DataFrame] = None
    abundance_df: Optional[pl.DataFrame] = None
    taxonomic_rank: str = "species"
    environmental_param: str = "site"
    beta_diversity_metric: str = "braycurtis"
    min_reads_per_sample: int = 100
    min_reads_per_taxon: int = 10
    permanova_permutations: int = 999
    species_filter: Optional[List[str]] = None
    sample_filter: Optional[List[str]] = None
    enable_clustering: bool = True
    clustering_method: str = "meanshift"
    output_prefix: str = "beta_diversity_analysis"
    save_intermediate: bool = False


@dataclass
class PipelineOutputs:
    """Output results from beta diversity pipeline."""

    results: Dict[str, Any]
    processed_data: Dict[str, Any]
    plots: List[Dict[str, Any]]
    cluster_info: Optional[Dict[str, Any]]
    saved_files: List[Path]
    performance_metrics: Dict[str, Any]
    execution_time: float


class BetaDiversityPipeline:
    """
    Complete beta diversity analysis pipeline.

    This class provides a comprehensive, production-ready implementation
    of beta diversity analysis that replaces the original beta.py script.
    """

    def __init__(self, config: Optional[BetaDiversityConfig] = None):
        """
        Initialize pipeline with configuration.

        Args:
            config: Optional configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        self.performance_logger = PerformanceLogger(self.logger)

        # Initialize components
        self.validator = DataValidator(self.config)
        self.processor = DataProcessor(self.config)
        self.analyzer = BetaDiversityAnalyzer(self.config)
        self.visualizer = BetaDiversityVisualizer(self.config)
        self.clustering_analyzer = ClusteringAnalyzer(self.config)
        self.storage_manager = StorageManager(self.config)

        self.logger.info("Beta diversity pipeline initialized")

    @performance_tracker("run_complete_pipeline")
    def run(self, inputs: PipelineInputs) -> PipelineOutputs:
        """
        Run complete beta diversity analysis pipeline.

        Args:
            inputs: Pipeline input parameters

        Returns:
            Complete pipeline outputs
        """
        start_time = time.time()

        try:
            self.logger.info("Starting beta diversity analysis pipeline")

            # Validate inputs
            self._validate_inputs(inputs)

            # Step 1: Load and process data
            with self.performance_logger.track_operation("data_processing"):
                processed_data = self._process_data(inputs)

            # Step 2: Run beta diversity analysis
            with self.performance_logger.track_operation("beta_diversity_analysis"):
                results = self._run_analysis(processed_data, inputs)

            # Step 3: Apply clustering if enabled
            cluster_info = None
            if inputs.enable_clustering:
                with self.performance_logger.track_operation("clustering"):
                    cluster_info = self._apply_clustering(results, inputs)

            # Step 4: Create visualizations
            with self.performance_logger.track_operation("visualization"):
                plots = self._create_visualizations(
                    results, processed_data, inputs, cluster_info
                )

            # Step 5: Save results
            with self.performance_logger.track_operation("save_results"):
                saved_files = self._save_results(results, processed_data, plots, inputs)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Get performance metrics
            performance_metrics = self.performance_logger.get_all_metrics()

            # Create output object
            outputs = PipelineOutputs(
                results=results,
                processed_data=processed_data,
                plots=plots,
                cluster_info=cluster_info,
                saved_files=saved_files,
                performance_metrics=performance_metrics,
                execution_time=execution_time,
            )

            self.logger.info(
                f"Pipeline completed successfully in {execution_time:.2f} seconds"
            )
            self.performance_logger.log_summary()

            return outputs

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise AnalysisError(f"Pipeline execution failed: {e}")

    def _validate_inputs(self, inputs: PipelineInputs) -> None:
        """Validate pipeline inputs."""
        # Check that either file paths or DataFrames are provided
        if inputs.metadata_path is None and inputs.metadata_df is None:
            raise DataValidationError(
                "Either metadata_path or metadata_df must be provided"
            )

        if inputs.abundance_path is None and inputs.abundance_df is None:
            raise DataValidationError(
                "Either abundance_path or abundance_df must be provided"
            )

        # Validate file paths exist if provided
        if inputs.metadata_path and not inputs.metadata_path.exists():
            raise DataValidationError(
                f"Metadata file not found: {inputs.metadata_path}"
            )

        if inputs.abundance_path and not inputs.abundance_path.exists():
            raise DataValidationError(
                f"Abundance file not found: {inputs.abundance_path}"
            )

        # Validate parameters
        if inputs.beta_diversity_metric not in self.config.analysis.supported_metrics:
            raise DataValidationError(
                f"Unsupported beta diversity metric: {inputs.beta_diversity_metric}"
            )

        if inputs.clustering_method not in self.config.analysis.supported_clustering:
            raise DataValidationError(
                f"Unsupported clustering method: {inputs.clustering_method}"
            )

        self.logger.info("Input validation completed")

    def _process_data(self, inputs: PipelineInputs) -> Dict[str, pl.DataFrame]:
        """Process input data for analysis."""
        self.logger.info("Processing input data")

        # Load and process metadata
        metadata = self.processor.load_and_process_metadata(
            metadata_path=inputs.metadata_path,
            metadata_df=inputs.metadata_df,
            environmental_param=inputs.environmental_param,
        )

        # Load and process abundance data
        abundance_data = self.processor.load_and_process_abundance_data(
            abundance_path=inputs.abundance_path,
            abundance_df=inputs.abundance_df,
            taxonomic_rank=inputs.taxonomic_rank,
            min_reads_per_sample=inputs.min_reads_per_sample,
            min_reads_per_taxon=inputs.min_reads_per_taxon,
        )

        # Prepare data for analysis
        merged_data, otu_matrix = self.processor.prepare_analysis_data(
            metadata=metadata,
            abundance_df=abundance_data,
            taxonomic_rank=inputs.taxonomic_rank,
            environmental_param=inputs.environmental_param,
            species_list=inputs.species_filter,
            sample_list=inputs.sample_filter,
        )

        processed_data = {
            "metadata": metadata,
            "abundance_data": abundance_data,
            "merged_data": merged_data,
            "otu_matrix": otu_matrix,
            "unique_metadata": merged_data.select(
                ["sample_id", inputs.environmental_param, "latitude", "longitude"]
            ).unique(subset=["sample_id"], keep="first"),
        }

        self.logger.info(f"Data processing completed: {len(processed_data)} datasets")

        return processed_data

    def _run_analysis(
        self, processed_data: Dict[str, pl.DataFrame], inputs: PipelineInputs
    ) -> BetaDiversityResults:
        """Run beta diversity analysis."""
        self.logger.info("Running beta diversity analysis")

        # Run complete analysis
        results = self.analyzer.run_complete_analysis(
            otu_matrix=processed_data["otu_matrix"],
            metadata=processed_data["unique_metadata"],
            taxonomic_rank=inputs.taxonomic_rank,
            environmental_param=inputs.environmental_param,
            metric=inputs.beta_diversity_metric,
            permutations=inputs.permanova_permutations,
        )

        self.logger.info("Beta diversity analysis completed")

        return results

    def _apply_clustering(
        self, results: BetaDiversityResults, inputs: PipelineInputs
    ) -> Dict[str, Any]:
        """Apply clustering analysis to PCoA results."""
        self.logger.info("Applying clustering analysis")

        # Get PCoA coordinates
        pc1_scores = results.sample_scores["PC1"]
        pc2_scores = results.sample_scores["PC2"]
        coordinates = np.column_stack([pc1_scores, pc2_scores])

        # Apply clustering
        (
            cluster_labels,
            unique_clusters,
            colors,
            method_info,
        ) = self.clustering_analyzer.apply_clustering(
            coordinates=coordinates, method=inputs.clustering_method
        )

        # Analyze cluster characteristics
        cluster_characteristics = (
            self.clustering_analyzer.analyze_cluster_characteristics(
                coordinates=coordinates,
                cluster_labels=cluster_labels,
                metadata=None,  # Could be enhanced to include metadata
            )
        )

        cluster_info = {
            "labels": cluster_labels,
            "unique_clusters": unique_clusters,
            "colors": colors,
            "method_info": method_info,
            "characteristics": cluster_characteristics,
        }

        self.logger.info(f"Clustering completed: {len(unique_clusters)} clusters found")

        return cluster_info

    def _create_visualizations(
        self,
        results: BetaDiversityResults,
        processed_data: Dict[str, pl.DataFrame],
        inputs: PipelineInputs,
        cluster_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create visualizations."""
        self.logger.info("Creating visualizations")

        plots = {}

        # Create main PCoA plot
        pcoa_plot = self.visualizer.create_pcoa_plot(
            results=results,
            metadata=processed_data["merged_data"],
            environmental_param=inputs.environmental_param,
        )

        # Add clustering ellipses if clustering was performed
        if cluster_info:
            pc1_scores = results.sample_scores["PC1"]
            pc2_scores = results.sample_scores["PC2"]
            coordinates = np.column_stack([pc1_scores, pc2_scores])

            pcoa_plot = self.visualizer.add_clustering_ellipses(
                fig=pcoa_plot,
                pcoa_scores=coordinates,
                cluster_labels=cluster_info["labels"],
                colors=cluster_info["colors"],
            )

        plots["pcoa_plot"] = pcoa_plot

        # Create distance matrix heatmap
        try:
            heatmap = self.visualizer.create_distance_matrix_heatmap(
                distance_matrix=results.distance_matrix,
                title="Beta Diversity Distance Matrix",
            )
            plots["distance_heatmap"] = heatmap
        except Exception as e:
            self.logger.warning(f"Failed to create distance matrix heatmap: {e}")

        # Create scree plot
        try:
            scree_plot = self.visualizer.create_scree_plot(
                pcoa_results=results.pcoa_results, title="PCoA Eigenvalues (Scree Plot)"
            )
            plots["scree_plot"] = scree_plot
        except Exception as e:
            self.logger.warning(f"Failed to create scree plot: {e}")

        self.logger.info(f"Visualization completed: {len(plots)} plots created")

        return plots

    def _save_results(
        self,
        results: BetaDiversityResults,
        processed_data: Dict[str, pl.DataFrame],
        plots: Dict[str, Any],
        inputs: PipelineInputs,
    ) -> Dict[str, Path]:
        """Save all results."""
        self.logger.info("Saving results")

        saved_files = {}

        # Save main results
        result_files = self.storage_manager.save_results(
            results=results,
            output_prefix=inputs.output_prefix,
            include_distance_matrix=True,
            include_plots=True,
        )
        saved_files.update(result_files)

        # Save processed data if requested
        if inputs.save_intermediate:
            data_files = self.storage_manager.save_data_frames(
                data_frames=processed_data,
                output_prefix=f"{inputs.output_prefix}_processed",
            )
            saved_files.update({f"data_{k}": v for k, v in data_files.items()})

        # Save plots
        for plot_name, plot_figure in plots.items():
            plot_files = self.storage_manager.save_plot_data(
                figure_data={"figure": plot_figure},
                output_prefix=f"{inputs.output_prefix}_{plot_name}",
            )
            saved_files.update(
                {f"plot_{plot_name}_{k}": v for k, v in plot_files.items()}
            )

        # Save comprehensive report
        report_metadata = {
            "inputs": {
                "taxonomic_rank": inputs.taxonomic_rank,
                "environmental_param": inputs.environmental_param,
                "beta_diversity_metric": inputs.beta_diversity_metric,
                "clustering_enabled": inputs.enable_clustering,
                "clustering_method": (
                    inputs.clustering_method if inputs.enable_clustering else None
                ),
            },
            "data_summary": {
                "n_samples": len(results.distance_matrix.ids),
                "n_taxa": processed_data["otu_matrix"].height,
                "environmental_param_unique_values": processed_data["merged_data"]
                .select(inputs.environmental_param)
                .unique()
                .height,
            },
        }

        report_file = self.storage_manager.save_analysis_report(
            results=results,
            metadata=report_metadata,
            output_prefix=inputs.output_prefix,
        )
        saved_files["analysis_report"] = report_file

        self.logger.info(f"Results saved: {len(saved_files)} files")

        return saved_files

    def run_batch_analysis(
        self, batch_inputs: List[PipelineInputs], parallel: bool = False
    ) -> List[PipelineOutputs]:
        """
        Run batch analysis on multiple datasets.

        Args:
            batch_inputs: List of pipeline input configurations
            parallel: Whether to run analyses in parallel

        Returns:
            List of pipeline outputs
        """
        try:
            self.logger.info(f"Starting batch analysis: {len(batch_inputs)} datasets")

            if parallel:
                return self._run_batch_parallel(batch_inputs)
            else:
                return self._run_batch_sequential(batch_inputs)

        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
            raise AnalysisError(f"Batch analysis failed: {e}")

    def _run_batch_sequential(
        self, batch_inputs: List[PipelineInputs]
    ) -> List[PipelineOutputs]:
        """Run batch analysis sequentially."""
        results = []
        progress = ProgressTracker(self.logger, len(batch_inputs), "Batch Analysis")

        for i, inputs in enumerate(batch_inputs):
            try:
                self.logger.info(
                    f"Processing dataset {i+1}/{len(batch_inputs)}: {inputs.output_prefix}"
                )
                output = self.run(inputs)
                results.append(output)
                progress.update(i + 1, f"Completed {inputs.output_prefix}")
            except Exception as e:
                self.logger.error(
                    f"Failed to process dataset {inputs.output_prefix}: {e}"
                )
                # Continue with other datasets
                results.append(None)
                progress.update(i + 1, f"Failed {inputs.output_prefix}")

        progress.finish("Batch analysis completed")

        return results

    def _run_batch_parallel(
        self, batch_inputs: List[PipelineInputs]
    ) -> List[PipelineOutputs]:
        """Run batch analysis in parallel."""
        import concurrent.futures

        max_workers = min(self.config.analysis.parallel_workers, len(batch_inputs))

        self.logger.info(f"Running batch analysis with {max_workers} workers")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all jobs
            future_to_input = {
                executor.submit(self._run_single_analysis, inputs): inputs
                for inputs in batch_inputs
            }

            results = []
            for future in concurrent.futures.as_completed(future_to_input):
                inputs = future_to_input[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Completed analysis: {inputs.output_prefix}")
                except Exception as e:
                    self.logger.error(
                        f"Analysis failed for {inputs.output_prefix}: {e}"
                    )
                    results.append(None)

        return results

    def _run_single_analysis(self, inputs: PipelineInputs) -> PipelineOutputs:
        """Run single analysis (for parallel execution)."""
        # Create new pipeline instance for parallel execution
        pipeline = BetaDiversityPipeline(self.config)
        return pipeline.run(inputs)

    def validate_pipeline_setup(self) -> Dict[str, Any]:
        """
        Validate that pipeline is properly set up.

        Returns:
            Validation results
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "component_status": {},
        }

        try:
            # Check component initialization
            components = {
                "validator": self.validator,
                "processor": self.processor,
                "analyzer": self.analyzer,
                "visualizer": self.visualizer,
                "clustering_analyzer": self.clustering_analyzer,
                "storage_manager": self.storage_manager,
            }

            for name, component in components.items():
                if component is None:
                    validation_results["errors"].append(
                        f"Component {name} not initialized"
                    )
                    validation_results["component_status"][name] = "failed"
                else:
                    validation_results["component_status"][name] = "ok"

            # Check configuration
            try:
                self.config.to_dict()  # Just validate, don't store
                validation_results["component_status"]["config"] = "ok"
            except Exception as e:
                validation_results["errors"].append(
                    f"Configuration validation failed: {e}"
                )
                validation_results["component_status"]["config"] = "failed"

            # Check storage directories
            try:
                storage_summary = self.storage_manager.get_storage_summary()
                validation_results["storage_summary"] = storage_summary
                validation_results["component_status"]["storage"] = "ok"
            except Exception as e:
                validation_results["warnings"].append(
                    f"Storage validation warning: {e}"
                )
                validation_results["component_status"]["storage"] = "warning"

            # Set overall status
            validation_results["valid"] = len(validation_results["errors"]) == 0

            status = 'PASSED' if validation_results['valid'] else 'FAILED'
            self.logger.info(f"Pipeline validation completed: {status}")

            return validation_results

        except Exception as e:
            self.logger.error(f"Pipeline validation failed: {e}")
            return {
                "valid": False,
                "errors": [f"Validation process failed: {e}"],
                "warnings": [],
                "component_status": {},
            }

    def create_inputs_from_files(
        self, metadata_path: Path, abundance_path: Path, **kwargs
    ) -> PipelineInputs:
        """
        Create pipeline inputs from file paths.

        Args:
            metadata_path: Path to metadata file
            abundance_path: Path to abundance data file
            **kwargs: Additional input parameters

        Returns:
            Pipeline inputs object
        """
        return PipelineInputs(
            metadata_path=metadata_path, abundance_path=abundance_path, **kwargs
        )

    def create_inputs_from_dataframes(
        self, metadata_df: pl.DataFrame, abundance_df: pl.DataFrame, **kwargs
    ) -> PipelineInputs:
        """
        Create pipeline inputs from DataFrames.

        Args:
            metadata_df: Metadata DataFrame
            abundance_df: Abundance DataFrame
            **kwargs: Additional input parameters

        Returns:
            Pipeline inputs object
        """
        return PipelineInputs(
            metadata_df=metadata_df, abundance_df=abundance_df, **kwargs
        )
