# Removed error_handler and log_upload for challenge simplicity
from shared.logger import info, debug, error
import time
import os
import json
import plotly.express as px
import plotly.utils
import polars as pl
from skbio import diversity
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa
import numpy as np
from alpha import CONTINUOUS_VARIABLES
from plotly.colors import sample_colorscale
from shared.clustering import apply_clustering
from shared.visualization import int_to_letter, point_on_ellipse
from shared.analysis import ticks_from_kmeans
from shared.text import humanize_parameter_name


# Custom JSON encoder to handle numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if obj != obj:  # NaN check without pandas
            return None
        return super().default(obj)


# from colormap import rgb2hex, hex2rgb  # Unused in current code
from math import cos, sin, pi
from datetime import datetime

from db.schema import Report, ComputeLog
from db.func import update_report_status
from db.enums import ReportBuildState
from db.session import start_db_session
# Import directly from stubs for simplicity
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "stubs"))

from metadata_stub import (
    process_metadata,
    load_reads_for_primer,
    get_species_list,
    get_labels,
    get_taxonomic_ranks
)
from shared.arguments import parse_arguments
from shared.labels import extract_legend_and_axis_label
from download.save_report_tarball import save_report_tarball
from storage.upload_from_bytes import upload_from_bytes
from compression.compress_string import compress_string
from progress.ProgressWebSocket import ProgressWebSocket


K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "staging")
BUCKET = os.getenv("GCS_BUCKET", "edna-project-files-{NAMESPACE}").replace(
    "{NAMESPACE}", K8S_NAMESPACE
)
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")


def beta(session: object):
    start_time = time.time()
    ws = ProgressWebSocket("example-report-id")
    ws.send_update("started", 0.0, "Report initialized")

    if not session:
        error("DB session not started. Exiting...")
        exit(1)

    # Example report data - would normally be fetched from database
    info("Fetching example report data")
    # report = session.query(Report).filter(Report.id == report_id).first()
    
    # Hardcoded report object for example purposes
    class MockReport:
        def __init__(self):
            self.taxonomicRank = 'species'
            self.confidenceLevel = 3
            self.firstDate = '2020-07-24'
            self.lastDate = '2021-02-21'
            self.id = 'example-report-id'
            self.project_id = 'example-project'
            self.sites = []
            self.tags = []
            self.environmentalParameter = 'site'
            self.speciesList = 'None'
            self.marker = 'example-marker'
            self.countThreshold = 10
            self.filterThreshold = 5
            self.betaDiversity = 'jaccard'
    
    report = MockReport()

    # Ensure that the taxonomic rank is in lowercase
    taxonomic_rank = report.taxonomicRank.lower()

    if report:
        update_report_status(session, report.id, ReportBuildState.QUEUED.value, True)
        session.commit()

        species_list_df = None
        if report.speciesList != "None":
            species_list_df = get_species_list(session, report.speciesList)

        r = process_metadata(
            session,
            project_id=report.project_id,
            filter_site_ids=report.sites,
            filter_tag_ids=report.tags,
            sample_first_date=report.firstDate,
            sample_last_date=report.lastDate,
            environmental_variable=report.environmentalParameter,
        )
        metadata = r.get("metadata")
        controls = r.get("controls")
        total_samples = r.get("total_samples")
        total_sites = r.get("total_sites")
        filtered_sites = r.get("filtered_sites")

        # Filter metadata
        metadata = metadata.filter(
            metadata["latitude"].is_not_null()
            & metadata["longitude"].is_not_null()
            & metadata["sample_id"].is_not_null()
        )

        # Assuming metadata is a DataFrame
        if len(metadata) == 0:
            exit("Error: Sample data frame is empty. Cannot proceed.")

        # Save a dataframe with the sites and samples for later use
        site_lookup = metadata.select("site", "sample_id").unique()

        # Keep metadata as Polars DataFrame

        if taxonomic_rank == "max":
            taxonomic_rank = "taxonomic_path"
        # Get taxonomic ranks and filter the prevalence
        [taxonomic_ranks, _, taxonomic_num] = get_taxonomic_ranks(
            taxonomic_rank=taxonomic_rank
        )

        # Download the primer file
        update_report_status(session, report.id, ReportBuildState.LOADING.value, False)
        session.commit()

        read_results = load_reads_for_primer(
            primer=report.marker,
            project_id=report.project_id,
            taxonomic_ranks=taxonomic_ranks,
            minimum_reads_per_sample=report.countThreshold,
            confidence=report.confidenceLevel,
            minimum_reads_per_taxon=report.filterThreshold,
            metadata=metadata,
            controls=controls["sample_id"].unique().to_list(),
            report_id="example-report-id",
        )

        tronko_db = read_results["decontaminated_reads"]
        sample_list = read_results["valid_samples"]

        labels_and_legend = get_labels(BUCKET, "analysis/LabelsAndLegends.csv")
        categories = get_labels(BUCKET, "analysis/Categories.csv")
        update_report_status(session, report.id, ReportBuildState.BUILDING.value, False)
        session.commit()

        if report.environmentalParameter == "iucn_cat":
            metadata = metadata.with_columns(
                pl.col("iucn_cat").fill_null("not reported")
            )

        # Adjust metadata with categories using Polars
        if (
            report.environmentalParameter in metadata.columns
            and report.environmentalParameter
            in categories["Environmental_Variable"].to_list()
        ):
            # Filter categories for the specific environmental variable
            relevant_categories = categories.filter(
                pl.col("Environmental_Variable") == report.environmentalParameter
            )

            # Rename the environmental_variable column in metadata to 'value'
            metadata = metadata.rename({report.environmentalParameter: "value"})

            # Merge metadata with categories on 'value'
            metadata = metadata.join(relevant_categories, on="value", how="left")

            # Rename the 'description' column back to the original environmental_variable name
            if "description" in metadata.columns:
                metadata = metadata.rename(
                    {"description": report.environmentalParameter}
                )

            print("Finished changing values of selected variable.")
        elif report.environmentalParameter in metadata.columns:
            metadata = metadata.with_columns(
                pl.col(report.environmentalParameter).alias("value")
            )
        else:
            print(
                f"{report.environmentalParameter} is not a column in the metadata DataFrame."
            )
        metadata = metadata.unique(subset=["sample_id"], keep="first")

        new_legend, _ = extract_legend_and_axis_label(
            labels_and_legend=labels_and_legend,
            environmental_variable=report.environmentalParameter,
        )

        # Remove taxa which are unknown at a given rank
        tronko_db = tronko_db.filter(tronko_db[taxonomic_rank].is_not_null())

        if species_list_df is not None:
            species_names = species_list_df.select(pl.col("name")).unique()
            tronko_db = tronko_db.filter((pl.col("species").is_in(species_names)))

        if sample_list is not None:
            tronko_db = tronko_db.filter(pl.col("sample_id").is_in(sample_list))

        # Create a combined taxonomic path column if needed
        if taxonomic_rank == "taxonomic_path":
            # Combine taxonomy columns into a single path
            tronko_db = tronko_db.with_columns(
                [
                    pl.concat_str(
                        [
                            pl.col("kingdom"),
                            pl.col("phylum"),
                            pl.col("class"),
                            pl.col("order"),
                            pl.col("family"),
                            pl.col("genus"),
                            pl.col("species"),
                        ],
                        separator=" > ",
                    ).alias("taxonomic_path")
                ]
            )

        # Include taxonomic_path in the columns if it's been created
        if taxonomic_rank == "taxonomic_path" and "taxonomic_path" in tronko_db.columns:
            tronko_input = tronko_db.select(
                ["sample_id", "freq", "taxonomic_path"]
                + taxonomic_ranks[0 : taxonomic_num + 1]
            )
        else:
            tronko_input = tronko_db.select(
                ["sample_id", "freq"] + taxonomic_ranks[0 : taxonomic_num + 1]
            )

        metadata = metadata.drop_nulls(subset=[report.environmentalParameter])

        tronko_input = tronko_input.join(
            metadata.select(["sample_id", report.environmentalParameter]),
            on="sample_id",
            how="left",
        ).drop_nulls(subset=[report.environmentalParameter])

        # Sort to ensure consistent ordering
        tronko_input = tronko_input.sort([taxonomic_rank, "sample_id"])
        valid_ids = tronko_input.select("sample_id").unique().to_series().to_list()

        ws.send_update("analyzing", 0.75, "Generating OTU matrix")

        if len(tronko_input) > 1:
            # Create OTU matrix with consistent ordering
            otumat = tronko_input.pivot(
                values="freq",
                index=taxonomic_rank,
                on="sample_id",
                aggregate_function="sum",
            ).fill_null(0)

            # Sort the matrix by taxonomic rank to ensure consistent row ordering
            otumat = otumat.sort(taxonomic_rank)

            # Get sample columns and sort them to ensure consistent column ordering
            sample_columns = [col for col in otumat.columns if col != taxonomic_rank]
            sample_columns.sort()  # Ensure consistent column order

            # Reorder columns: taxonomic_rank first, then sorted sample columns
            otumat = otumat.select([taxonomic_rank] + sample_columns)

            # Ensure all data columns (except index) are numeric
            otumat = otumat.with_columns(
                [pl.col(col).cast(pl.Float64) for col in sample_columns]
            )

            if otumat.shape[1] > 1:

                # Create merged Phyloseq object (skbio doesn't have Phyloseq, but you can use its diversity measures)
                # Just using a single beta diversity metric 'jaccard' as an example
                beta_diversity_metric = (
                    "braycurtis"
                    if report.betaDiversity == "bray"
                    else report.betaDiversity
                )
                info(f"Using beta diversity metric: {beta_diversity_metric}")

                ws.send_update("analyzing", 0.8, "Calculating distance matrices")
                debug("Calculating beta diversity")
                # Calculate beta diversity matrix
                # Get column names (sample IDs) excluding the taxonomic rank column
                sample_ids = [col for col in otumat.columns if col != taxonomic_rank]
                sample_ids.sort()  # Ensure consistent ordering for scikit-bio
                otumat_values = otumat.select(sample_ids).to_numpy()
                beta_diversity = diversity.beta_diversity(
                    beta_diversity_metric, otumat_values.T, ids=sample_ids
                )

                # Check if the distance matrix is valid for PCoA
                if (
                    np.isnan(beta_diversity.data).any()
                    or np.isinf(beta_diversity.data).any()
                ):
                    update_report_status(
                        session,
                        "example-report-id",
                        ReportBuildState.FAILED.value,
                        False,
                        "Invalid distance matrix computed. Please try a different beta diversity metric.",
                        "error",
                    )
                    session.commit()
                    return

                # Perform PCoA on the beta diversity Distance Matrix
                try:
                    ordination_results = pcoa(beta_diversity)
                except Exception as e:
                    update_report_status(
                        session,
                        "example-report-id",
                        ReportBuildState.FAILED.value,
                        False,
                        f"PCoA calculation failed: {str(e)}",
                        "error",
                    )
                    session.commit()
                    return
                pc1_scores = ordination_results.samples["PC1"]
                pc2_scores = ordination_results.samples["PC2"]

                # Assuming ordination_results has an attribute for eigenvalues or a similar metric
                # For PCoA in scikit-bio, eigenvalues can be accessed via ordination_results.eigvals for example

                # Calculate the sum of all eigenvalues
                total_variance = sum(ordination_results.eigvals)

                # Calculate the variance explained by PC1 and PC2
                variance_explained_pc1 = (
                    ordination_results.eigvals["PC1"] / total_variance
                )
                variance_explained_pc2 = (
                    ordination_results.eigvals["PC2"] / total_variance
                )

                # Convert to percentage
                percentage_explained_pc1 = variance_explained_pc1 * 100
                percentage_explained_pc2 = variance_explained_pc2 * 100

                # Convert beta_diversity to DataFrame with sorted sample IDs
                beta_df = pl.DataFrame(
                    {
                        "sample_id": list(beta_diversity.ids),
                        "beta_diversity": list(beta_diversity.data),
                    }
                ).sort("sample_id")

                # Merge metadata with beta diversity
                # Sort metadata by sample_id to ensure consistent ordering
                metadata_sorted = metadata.sort("sample_id")
                merged_df = beta_df.join(metadata_sorted, on="sample_id", how="left")

                # Sort final merged dataframe by sample_id to ensure consistent ordering
                merged_df = merged_df.sort("sample_id")
                info("Successfully performed beta diversity calculation")

                # Debug: Check temporal_months data after processing
                if report.environmentalParameter == "temporal_months":
                    debug(f"DEBUG temporal_months: After metadata processing")
                    debug(
                        f"temporal_months unique values: {merged_df.select('temporal_months').unique().sort('temporal_months').to_series().to_list()}"
                    )
                    debug(f"temporal_months sample count: {len(merged_df)}")
                    debug(
                        f"temporal_months null count: {merged_df.filter(pl.col('temporal_months').is_null()).height}"
                    )

                # Add PCoA scores to your merged DataFrame
                # Convert PC scores to Polars DataFrame and join
                pc_scores_df = pl.DataFrame(
                    {
                        "sample_id": list(pc1_scores.index),
                        "PC1": list(pc1_scores.values),
                        "PC2": list(pc2_scores.values),
                    }
                ).sort("sample_id")
                merged_df = merged_df.join(pc_scores_df, on="sample_id", how="left")

                # Perform PERMANOVA
                # Check if there's enough variation in the environmental variable
                if len(merged_df.select(report.environmentalParameter).unique()) > 1:
                    ws.send_update("analyzing", 0.9, "Performing PERMANOVA")

                    # After filtering the DataFrame
                    filtered_df = merged_df.drop_nulls(
                        subset=[report.environmentalParameter]
                    ).sort(
                        "sample_id"
                    )  # Ensure consistent ordering
                    valid_ids = (
                        filtered_df.select("sample_id").unique().to_series().to_list()
                    )
                    valid_ids.sort()  # Sort valid_ids for consistent distance matrix filtering

                    # Debug prints
                    debug("\nDetailed ID verification:")
                    debug(f"Shape of filtered DataFrame: {filtered_df.shape}")
                    debug(
                        f"Number of unique sample IDs in filtered DataFrame: {len(filtered_df.select('sample_id').unique())}"
                    )
                    debug(
                        f"First few rows of filtered_df['sample_id']:\n{filtered_df.select('sample_id').head()}"
                    )

                    # Filter distance matrix
                    filtered_beta_diversity_dm = beta_diversity.filter(valid_ids)

                    # Create aligned DataFrame step by step
                    # Create a lookup dictionary from filtered_df for the alignment logic
                    temp_dict = dict(
                        zip(
                            filtered_df.select("sample_id").to_series().to_list(),
                            filtered_df.select(report.environmentalParameter)
                            .to_series()
                            .to_list(),
                        )
                    )
                    aligned_grouping = []
                    for sample_id in filtered_beta_diversity_dm.ids:
                        aligned_grouping.append(temp_dict[sample_id])
                        # value = temp_df.loc[sample_id][report.environmentalParameter]
                        # # Ensure temporal variables are properly handled as numeric types
                        # if report.environmentalParameter in ["temporal_months", "temporal_years", "temporal_days"]:
                        #     value = pd.to_numeric(value, errors='coerce')
                        # aligned_grouping.append(value)

                    filtered_df_aligned = pl.DataFrame(
                        {
                            "sample_id": list(filtered_beta_diversity_dm.ids),
                            "grouping": aligned_grouping,
                        }
                    )

                    debug("\nAfter alignment:")
                    debug(f"Distance matrix IDs: {len(filtered_beta_diversity_dm.ids)}")
                    debug(f"DataFrame samples: {len(filtered_df_aligned)}")
                    debug(
                        f"Are all distance matrix IDs in DataFrame? {all(id_ in filtered_df_aligned.select('sample_id').to_series().to_list() for id_ in filtered_beta_diversity_dm.ids)}"
                    )

                    # Create grouping array for PERMANOVA
                    # PERMANOVA expects a simple array/list of grouping values
                    grouping_values = (
                        filtered_df_aligned.select("grouping").to_series().to_list()
                    )

                    # Now perform PERMANOVA with the aligned grouping
                    permanova_results = permanova(
                        distance_matrix=filtered_beta_diversity_dm,
                        grouping=grouping_values,
                        permutations=999,
                    )

                    ws.send_update("analyzing", 0.9, "Plotting results")
                    plot_title = (
                        f"PCoA plot. Results of PERMANOVA, using {permanova_results['number of permutations']} permutations.\n"
                        f"{report.betaDiversity} beta diversity and {new_legend if new_legend is not None else humanize_parameter_name(report.environmentalParameter)}\n"
                        f"Sample size: {permanova_results['sample size']}. "
                        f"Number of groups: {permanova_results['number of groups']}. "
                        f"Test statistic (pseudo-F): {permanova_results['test statistic']:.3f}. "
                        f"p-value: {permanova_results['p-value']:.3f}"
                    )

                    X = merged_df.select(["PC1", "PC2"]).to_numpy()

                    # Apply clustering - can switch between "meanshift", "optics", and "balanced"
                    # Check environment variable or default to balanced approach
                    clustering_method = os.getenv(
                        "CLUSTERING_METHOD", "meanshift"
                    ).lower()
                    
                    if clustering_method not in ["meanshift", "optics", "balanced"]:
                        clustering_method = "meanshift"  # Default fallback

                    cluster_labels, unique_clusters, colors, method_info = (
                        apply_clustering(X, method=clustering_method)
                    )

                    # Add cluster labels to DataFrame if clustering was successful
                    if len(unique_clusters) > 0:
                        merged_df = merged_df.with_columns(
                            pl.Series("Cluster", cluster_labels)
                        )

                    # Set color scale based on environmental parameter
                    color_scale = None
                    color_map = None
                    continuous = report.environmentalParameter in CONTINUOUS_VARIABLES
                    if continuous:
                        color_scale = px.colors.sequential.Viridis
                        merged_df = merged_df.with_columns(
                            pl.col(report.environmentalParameter).cast(
                                pl.Float64, strict=False
                            )
                        )
                    else:
                        sites = sorted(
                            merged_df.select(report.environmentalParameter)
                            .unique()
                            .to_series()
                            .to_list()
                        )
                        n = len(sites)

                        palette = sample_colorscale(
                            "Turbo", [i / (n - 1) for i in range(n)]
                        )  # hex list
                        color_map = dict(zip(sites, palette))

                        # Debug: Color mapping for categorical variables
                        if report.environmentalParameter == "temporal_months":
                            debug(f"DEBUG temporal_months: Color mapping")
                            debug(f"Categorical sites/values: {sites}")
                            debug(f"Number of unique values: {n}")
                            debug(f"Color palette: {palette}")
                            debug(f"Color map: {color_map}")
                    # Include environmental parameter in hover_data for continuous variables
                    hover_data_cols = ["site", "sample_id"]
                    if continuous:
                        hover_data_cols.append(report.environmentalParameter)
                    # Visualization with Plotly (e.g., a scatter plot of the first two PCoA axes)
                    # Convert to dictionary format for Plotly compatibility
                    merged_df_dict = merged_df.to_dicts()

                    # Debug: Check data before plotly
                    if report.environmentalParameter == "temporal_months":
                        debug(f"DEBUG temporal_months: Before plotly figure creation")
                        sample_values = [
                            d.get(report.environmentalParameter)
                            for d in merged_df_dict[:5]
                        ]
                        debug(f"Sample temporal_months values in dict: {sample_values}")
                        debug(f"Using color_discrete_map: {color_map}")

                    fig = px.scatter(
                        merged_df_dict,
                        x="PC1",
                        y="PC2",
                        color=report.environmentalParameter,
                        labels={
                            "PC1": f"PCoA 1 ({percentage_explained_pc1:.2f}%)",
                            "PC2": f"PCoA 2 ({percentage_explained_pc2:.2f}%)",
                            "site": "Site",
                            "sample_id": "Sample ID",
                        },
                        hover_data=hover_data_cols,
                        color_continuous_scale=color_scale,
                        color_discrete_map=color_map,
                    )

                    # Debug: Check figure traces after creation
                    if report.environmentalParameter == "temporal_months":
                        debug(f"DEBUG temporal_months: After plotly figure creation")
                        debug(f"Number of traces: {len(fig.data)}")
                        for i, trace in enumerate(fig.data):
                            trace_name = (
                                trace.name if hasattr(trace, "name") else "NO_NAME"
                            )
                            debug(
                                f"Trace {i}: name='{trace_name}', type={type(trace).__name__}"
                            )
                            if hasattr(trace, "marker"):
                                debug(
                                    f"  marker.color: {getattr(trace.marker, 'color', 'NO_COLOR')}"
                                )
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0, pad=0))
                    fig.update_traces(marker={"symbol": "square"})
                    # Update hover template to show site and sample ID (avoid duplicating site if that's the environmental parameter)
                    for i in range(len(fig.data)):
                        current_name = (
                            fig.data[i].name if hasattr(fig.data[i], "name") else ""
                        )

                        # Determine if we need to show the site (skip if the environmental parameter is "site")
                        site_template = (
                            ""
                            if report.environmentalParameter == "site"
                            else "<b>Site:</b> %{customdata[0]}<br>"
                        )

                        # For continuous variables, show the actual value instead of the categorical name
                        legend_label = (
                            new_legend
                            if new_legend is not None
                            else humanize_parameter_name(report.environmentalParameter)
                        )
                        if continuous:
                            env_param_template = (
                                f"<b>{legend_label}:</b> %{{customdata[2]}}<br>"
                            )
                        else:
                            env_param_template = (
                                f"<b>{legend_label}:</b> {current_name}<br>"
                            )
                        fig.data[i].hovertemplate = (
                            f"{site_template}"
                            "<b>Sample ID:</b> %{customdata[1]}<br>"
                            f"{env_param_template}"
                            "<b>PCoA 1:</b> %{x:.4f}<br>"
                            "<b>PCoA 2:</b> %{y:.4f}<br>"
                            "<extra></extra>"
                        )
                    x_min, x_max = (
                        merged_df.select("PC1").min().item(),
                        merged_df.select("PC1").max().item(),
                    )
                    y_min, y_max = (
                        merged_df.select("PC2").min().item(),
                        merged_df.select("PC2").max().item(),
                    )
                    x_range = x_max - x_min
                    y_range = y_max - y_min

                    # Convert the beta_diversity column to a string representation
                    pl_df = merged_df
                    pl_df = pl_df.with_columns(
                        pl.col("beta_diversity")
                        .map_elements(
                            lambda x: ",".join(map(str, x)), return_dtype=pl.String
                        )
                        .alias("beta_diversity_str")
                    ).drop("beta_diversity")

                    if len(unique_clusters) > 0:
                        pl_df = pl_df.with_columns(
                            pl.col("Cluster").alias("community"),
                        ).drop("Cluster")

                    # Rename the new string column back to beta_diversity
                    pl_df = pl_df.rename({"beta_diversity_str": "beta_diversity"})

                    # Sort by sample_id to ensure consistent output ordering
                    pl_df = pl_df.sort("sample_id")

                    # Convert date columns to strings with original format to match pandas output
                    date_columns = [
                        col for col in pl_df.columns if pl_df.schema[col] == pl.Date
                    ]
                    if date_columns:
                        pl_df = pl_df.with_columns(
                            [
                                (
                                    pl.col(col).dt.strftime("%Y-%m-%d")
                                    + "T00:00:00.000"
                                ).alias(col)
                                for col in date_columns
                            ]
                        )

                    pl_df.write_csv("beta_diversity.tsv", separator="\t")

                    if len(unique_clusters) > 0:
                        # Calculate and add bounding circles for each cluster
                        for i, cluster_id in enumerate(unique_clusters):
                            color = colors[
                                i % len(colors)
                            ]  # Use modulo to cycle through colors if more clusters than colors
                            cluster_points = X[cluster_labels == cluster_id]

                            # Calculate center of the cluster
                            center_x, center_y = np.mean(cluster_points, axis=0)
                            # Calculate and check covariance matrix
                            cov = np.cov(cluster_points.T)

                            # Check if cluster has enough points for meaningful covariance
                            if (
                                len(cluster_points) <= 1
                                or np.isnan(cov).any()
                                or np.isinf(cov).any()
                            ):
                                print(
                                    f"Skipping cluster {cluster_id} due to insufficient data or invalid covariance"
                                )
                                continue

                            # Calculate eigenvalues and eigenvectors
                            try:
                                eigvals, eigvecs = np.linalg.eig(cov)
                                print(f"Eigenvalues: {eigvals}")
                            except Exception as e:
                                print(
                                    f"Error calculating eigenvalues for cluster {cluster_id}: {e}"
                                )
                                continue

                            # Sort eigenvalues and eigenvectors
                            order = eigvals.argsort()[::-1]
                            eigvals, eigvecs = eigvals[order], eigvecs[:, order]

                            # Calculate angle of rotation
                            angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

                            # Calculate width and height of ellipse (using 2 standard deviations)
                            # Ensure eigvals are positive to avoid sqrt of negative values
                            eigvals_positive = np.abs(
                                eigvals
                            )  # Use absolute values to avoid sqrt errors
                            print(f"Using eigenvalues (abs): {eigvals_positive}")
                            width, height = 4 * np.sqrt(eigvals_positive)
                            print(f"Ellipse width, height: {width}, {height}")

                            # Generate points for the ellipse
                            t = np.linspace(0, 2 * pi, 100)
                            ellipse_x = width / 2 * np.cos(t)
                            ellipse_y = height / 2 * np.sin(t)

                            # Rotate the ellipse
                            R = np.array(
                                [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
                            )
                            rotated = np.dot(R, np.array([ellipse_x, ellipse_y]))

                            # Translate the ellipse to the center of the cluster
                            x = rotated[0, :] + center_x
                            y = rotated[1, :] + center_y

                            # Create the path for the ellipse
                            path = f"M {x[0]},{y[0]}"
                            for i in range(1, len(x)):
                                path += f" L{x[i]},{y[i]}"
                            path += " Z"

                            # Add custom path shape
                            fig.add_shape(
                                type="path", path=path, line_color=color, line_width=4
                            )

                            # Calculate point for circle and label (e.g., at 45 degrees)
                            label_angle = np.pi / 4  # 45 degrees
                            circle_x, circle_y = point_on_ellipse(
                                center_x, center_y, width, height, angle, label_angle
                            )

                            # Calculate the radius for the background circle
                            radius = min(x_range, y_range) / 20

                            # Add solid circle behind the annotation
                            fig.add_shape(
                                type="circle",
                                xref="x",
                                yref="y",
                                x0=circle_x - radius,
                                y0=circle_y - radius,
                                x1=circle_x + radius,
                                y1=circle_y + radius,
                                fillcolor=color,
                                line_color=color,
                                opacity=1,
                            )

                            # Add cluster label
                            fig.add_annotation(
                                x=circle_x,
                                y=circle_y,
                                text=f"{int_to_letter(cluster_id+1)}",
                                showarrow=False,
                                font=dict(family="Arial", size=18, color="white"),
                                opacity=1,
                                align="center",
                            )

                            fig.update_layout(
                                yaxis=dict(
                                    scaleanchor="x",
                                    scaleratio=1,
                                ),
                                showlegend=False,
                            )

                    debug("Generated plotly plot")

                    # Initialize a dictionary to store sample counts
                    metadata = {
                        "totalSamples": 0,
                        "filteredSamples": 0,
                        "environmentalParameter": report.environmentalParameter,
                        "environmentalParameterLabel": (
                            new_legend
                            if new_legend is not None
                            else humanize_parameter_name(report.environmentalParameter)
                        ),
                        "betaDiversity": report.betaDiversity,
                    }

                    # Add legend data to metadata
                    metadata["legend"] = None
                    metadata["colorbar"] = None

                    if not continuous:
                        legend_data = []
                        for trace in fig.data:
                            legend_data.append(
                                {
                                    "name": trace["name"],
                                    "color": (
                                        trace["marker"]["color"]
                                        if "marker" in trace
                                        else None
                                    ),
                                    "symbol": (
                                        trace["marker"]["symbol"]
                                        if "marker" in trace
                                        else None
                                    ),
                                }
                            )
                        metadata["legend"] = legend_data

                    else:  # ← Plotly Express path
                        # Debug: Print all coloraxis and colorbar information
                        debug(
                            "=== DEBUG: Inspecting fig.layout for coloraxis/colorbar ==="
                        )
                        layout_attrs = dir(fig.layout)
                        coloraxis_attrs = [
                            attr for attr in layout_attrs if "coloraxis" in attr.lower()
                        ]
                        colorbar_attrs = [
                            attr for attr in layout_attrs if "colorbar" in attr.lower()
                        ]

                        debug(f"Coloraxis attributes found: {coloraxis_attrs}")
                        debug(f"Colorbar attributes found: {colorbar_attrs}")

                        # Check each coloraxis attribute
                        for attr in coloraxis_attrs:
                            value = getattr(fig.layout, attr, None)
                            if value is not None:
                                debug(f"fig.layout.{attr}: {value}")

                        # Check traces for coloraxis references
                        debug("=== Traces with coloraxis ===")
                        for i, trace in enumerate(fig.data):
                            if hasattr(trace, "marker") and hasattr(
                                trace.marker, "coloraxis"
                            ):
                                debug(
                                    f"Trace {i}: marker.coloraxis = {trace.marker.coloraxis}"
                                )

                        debug("=== End Debug ===")

                        # --- pick *one* trace that is being coloured -------------------------------
                        trace = next(
                            t
                            for t in fig.data  # e.g. the first marker trace
                            if getattr(t, "marker", None)  # has a marker object
                            and getattr(
                                t.marker, "coloraxis", None
                            )  # and that marker uses a colour-axis
                        )

                        axis_id = trace.marker.coloraxis  # 'coloraxis', 'coloraxis2', …

                        # --- grab the matching colour-axis object from the layout -------------------
                        ca = getattr(fig.layout, axis_id)  # or:  fig.layout[axis_id]
                        ca = fig.layout.coloraxis  # layout.coloraxis object
                        cbar = ca.colorbar  # layout.coloraxis.colorbar

                        min_val = ca.cmin
                        max_val = ca.cmax
                        if not min_val or not max_val:
                            min_val = min(fig.data[0].marker.color)
                            max_val = max(fig.data[0].marker.color)

                        tickvals = getattr(cbar, "tickvals", None)
                        ticktext = getattr(cbar, "ticktext", None)
                        if tickvals is None:
                            # Generate optimal ticks using KMeans clustering
                            color_values = np.asarray(
                                fig.data[0].marker.color, dtype=float
                            )
                            tickvals, ticktext = ticks_from_kmeans(
                                color_values, k_min=3, k_max=7
                            )

                        debug(f"This trace uses {axis_id}")
                        debug(
                            f"Colourscale : {ca.colorscale}"
                        )  # the scale actually applied
                        debug(
                            f"cmin/cmax   : {min_val}, {max_val}"
                        )  # the numeric domain
                        debug(f"tickvals    : {tickvals}")
                        debug(f"ticktext    : {ticktext}")

                        metadata["colorbar"] = {
                            # "title"      : getattr(cbar.title, "text", None),
                            "range": [min_val, max_val],  # numeric domain
                            "colorscale": list(ca.colorscale),  # [[frac, "rgb()"], …]
                            "tickvals": tickvals,
                            "ticktext": ticktext,
                        }

                    if metadata["colorbar"]:
                        # pull the numeric colour values that PX stored on the first trace
                        vals = np.asarray(fig.data[0].marker.color, dtype=float)

                        vmin, vmax = vals.min(), vals.max()  # domain
                        n_ticks = 6  # whatever you like

                        tickvals = np.linspace(vmin, vmax, n_ticks)  # linear ticks
                        # or:  tickvals = np.percentile(vals, [0,25,50,75,100])   # quantile ticks

                        ticktext = [f"{v:.2f}" for v in tickvals]

                        metadata["colorbar"].update(
                            tickvals=tickvals.tolist(),  # JSON-serialisable
                            ticktext=ticktext,
                        )

                    # Hide color scale
                    fig.update_layout(coloraxis_showscale=False)

                    # Use consistent sample counting - total from metadata, filtered from actual analysis
                    metadata["totalSamples"] = total_samples
                    metadata["filteredSamples"] = len(
                        merged_df.select("sample_id").unique()
                    )

                    filtered_sites = (
                        tronko_db.select("sample_id")
                        .unique()
                        .join(
                            site_lookup,
                            left_on="sample_id",
                            right_on="sample_id",
                            how="inner",
                        )
                        .select("site")
                        .unique()
                    )

                    metadata["legendTitle"] = (
                        new_legend
                        if new_legend is not None
                        else humanize_parameter_name(report.environmentalParameter)
                    )
                    metadata["filteredSites"] = filtered_sites.height
                    metadata["totalSites"] = total_sites

                    # Add PERMANOVA metadata to metadata
                    if permanova_results.any():
                        # Convert NumPy arrays to lists for JSON serialization
                        permanova_dict = {}
                        for key, value in permanova_results.items():
                            if hasattr(value, "tolist") and callable(value.tolist):
                                permanova_dict[key] = value.tolist()
                            else:
                                permanova_dict[key] = value
                        metadata["permanova"] = permanova_dict

                    # Add beta diversity information to metadata
                    metadata["beta_diversity"] = {
                        "metric": report.betaDiversity,
                    }

                    # Add cluster metadata to metadata
                    cluster_metadata = []
                    for i, cluster in enumerate(unique_clusters):
                        cluster_data = (
                            merged_df.filter(pl.col("Cluster") == cluster)
                            .select(["site", "latitude", "longitude", "sample_id"])
                            .sort("sample_id")
                        )
                        # Aggregate to get the centroid of latitude and longitude for each site
                        aggregated_site_data = (
                            cluster_data.group_by("site")
                            .agg(
                                [pl.col("latitude").mean(), pl.col("longitude").mean()]
                            )
                            .sort("site")
                        )
                        cluster_metadata.append(
                            {
                                "name": int_to_letter(cluster + 1),
                                "color": colors[i % len(colors)],
                                "sites": aggregated_site_data.select(
                                    ["site", "latitude", "longitude"]
                                )
                                .unique()
                                .to_dicts(),
                                "samples": cluster_data.unique().to_dicts(),
                            }
                        )
                    metadata["clusters"] = cluster_metadata

                    plotly_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    legend_label = (
                        new_legend
                        if new_legend is not None
                        else humanize_parameter_name(report.environmentalParameter)
                    )
                    updated_plotly_json_str = plotly_json.replace(
                        f'"{report.environmentalParameter}"', f'"{legend_label}"'
                    )

                    datasets = {
                        "results": updated_plotly_json_str,
                        "metadata": metadata,
                    }
                    file_contents = json.dumps({"datasets": datasets}, cls=NumpyEncoder)

                    debug(f"Uploading report JSON")
                    # Upload report JSON
                    object_key = f"projects/{report.project_id}/cache/{report.marker}/reports/{report.id}.json.zst"
                    compressed_file = compress_string(file_contents)
                    upload_from_bytes(
                        bucket_name=BUCKET,
                        target_path=object_key,
                        bytes=compressed_file,
                    )

                    # Upload tarball with TSV output
                    tsv_data = (
                        merged_df.sort("sample_id")  # Ensure consistent ordering
                        .drop("beta_diversity")
                        .rename({"PC1": "PCoA1", "PC2": "PCoA2"})
                        .with_columns(
                            [
                                pl.lit(f"{percentage_explained_pc1:.2f}%").alias(
                                    "% explained PCoA1"
                                ),
                                pl.lit(f"{percentage_explained_pc2:.2f}%").alias(
                                    "% explained PCoA2"
                                ),
                            ]
                        )
                    )

                    # Convert date columns to strings with original format for tarball
                    date_columns = [
                        col
                        for col in tsv_data.columns
                        if tsv_data.schema[col] == pl.Date
                    ]
                    if date_columns:
                        tsv_data = tsv_data.with_columns(
                            [
                                (
                                    pl.col(col).dt.strftime("%Y-%m-%d")
                                    + "T00:00:00.000"
                                ).alias(col)
                                for col in date_columns
                            ]
                        )
                    save_report_tarball(
                        [("beta", tsv_data)],
                        report,
                        BUCKET,
                        ["-----------------------------", plot_title],
                    )

                    # Export plotly visualization as HTML for review
                    os.makedirs("previews", exist_ok=True)
                    output_html_path = "previews/beta_diversity_report.html"
                    fig.write_html(output_html_path)
                    info(f"Plotly visualization saved to: {output_html_path}")
                    print(f"\n🎯 Beta diversity report HTML saved to: {output_html_path}")
                    print("📊 Open this file in your browser to view the interactive visualization")

                    ws.send_update("completed", 1.0, "Report completed")
                    ws.close()
                    update_report_status(
                        session, report.id, ReportBuildState.COMPLETED.value, False
                    )
                    # Database update removed for mock data challenge
                    # session.query(Report).filter(Report.id == report.id).update(
                    #     {"fileKey": object_key}, synchronize_session="fetch"
                    # )
                    session.commit()
                else:
                    print(
                        "Not enough variation in the environmental variable to perform a PERMANOVA."
                    )
                    update_report_status(
                        session,
                        "example-report-id",
                        ReportBuildState.FAILED.value,
                        False,
                        f"Not enough variation in the environmental variable to perform a PERMANOVA. All sample locations have the same {report.environmentalParameter} value of '{merged_df.select(report.environmentalParameter).unique().to_series().to_list()[0]}'.",
                        "error",
                    )
                    session.commit()
            else:
                update_report_status(
                    session,
                    "example-report-id",
                    ReportBuildState.FAILED.value,
                    False,
                    "There is not enough data in this project to calculate beta diversity for the current filter settings.",
                    "error",
                )
                session.commit()
        else:
            update_report_status(
                session,
                "example-report-id",
                ReportBuildState.FAILED.value,
                False,
                "There is not enough data in this project to calculate beta diversity using the specified environmental variable.",
                "error",
            )
            session.commit()

    compute = ComputeLog(
        project_id=report.project_id,
        description="beta_diversity_report",
        operation="report",
        executedAt=datetime.fromtimestamp(start_time),
        duration=int((time.time() - start_time) * 1000),
        cores=8,
        memory=8,
    )
    session.add(compute)
    session.close()

    # Calculate and log the total execution time
    end_time = time.time()  # Capture the end time
    total_time = end_time - start_time
    info(f"Total execution time: {total_time:.2f} seconds")


def main():
    session = start_db_session(K8S_NAMESPACE)

    try:
        info("Starting the beta script...")
        beta(session=session)
    except Exception as e:
        print(f"Error in beta diversity calculation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    main()
