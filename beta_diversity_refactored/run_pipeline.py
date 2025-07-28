#!/usr/bin/env python3
"""
Main entry point to run the refactored beta diversity pipeline.
"""

import sys
import logging
from pathlib import Path

# Add the parent directory to path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from beta_diversity_refactored.pipeline import BetaDiversityPipeline, PipelineInputs
from beta_diversity_refactored.config import get_config


def main():
    """
    Run the beta diversity pipeline with test data.
    """
    print("Starting beta diversity pipeline...")

    # Get configuration and enable fast mode for performance
    config = get_config()
    config.analysis.fast_mode = True
    config.validation.fast_mode = True

    # Set up logging with optimized level for fast mode
    log_level = logging.WARNING if config.analysis.fast_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize pipeline
    pipeline = BetaDiversityPipeline(config)

    # Set up test data paths
    test_data_dir = Path(__file__).parent.parent / "test_data"

    # Create pipeline inputs - using same data as original script
    inputs = PipelineInputs(
        metadata_path=test_data_dir / "sample_metadata.csv",
        abundance_path=test_data_dir / "decontaminated_reads.csv",
        taxonomic_rank="species",
        environmental_param="site",
        beta_diversity_metric="jaccard",  # Match original script
        min_reads_per_sample=10,  # Match original script (countThreshold)
        min_reads_per_taxon=5,  # Match original script (filterThreshold)
        permanova_permutations=999,  # Match original script
        enable_clustering=True,
        clustering_method="meanshift",  # Match original script
        output_prefix="full_beta_diversity",
        save_intermediate=False,  # Disable intermediate saves for performance
    )

    try:
        print("Input data paths:")
        print(f"  Metadata: {inputs.metadata_path}")
        print(f"  Abundance: {inputs.abundance_path}")
        print(
            "  Both paths exist: "
            f"{inputs.metadata_path.exists() and inputs.abundance_path.exists()}"
        )

        # Run the pipeline
        print("Running pipeline...")
        results = pipeline.run(inputs)

        print("Pipeline completed successfully!")
        print(f"Execution time: {results.execution_time:.2f} seconds")
        print("Results summary:")

        # Get beta diversity matrix safely
        distance_matrix = getattr(results.results, "distance_matrix", None)
        if distance_matrix is not None:
            print(f"  - Beta diversity matrix shape: {distance_matrix.shape}")

        # Get PERMANOVA results safely
        permanova_results = getattr(results.results, "permanova_results", {})
        if permanova_results:
            # Try different possible keys for p-value
            p_value = (
                permanova_results.get("p_value")
                or permanova_results.get("p-value")
                or permanova_results.get("p")
                or "N/A"
            )
        else:
            p_value = "N/A"
        print(f"  - PERMANOVA p-value: {p_value}")

        print(f"  - Number of plots generated: {len(results.plots)}")
        print(f"  - Saved files: {list(results.saved_files.keys())}")

        if results.cluster_info:
            method_info = results.cluster_info.get("method_info", {})
            print(f"  - Clustering method: {method_info.get('method', 'N/A')}")
            print("  - Number of clusters: " f"{method_info.get('n_clusters', 'N/A')}")

        return 0

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
