# Removed error_handler and log_upload for challenge simplicity
import os
import time
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from scipy import stats
import pandas as pd
import polars as pl
from skbio import diversity
import numpy as np
import statsmodels.api as sm
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
from shared.labels import extract_legend_and_axis_label, adjust_metadata_with_categories
from download.save_report_tarball import save_report_tarball
from storage.upload_from_bytes import upload_from_bytes
from progress.ProgressWebSocket import ProgressWebSocket
from compression.compress_string import compress_string


K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "staging")
BUCKET = os.getenv("GCS_BUCKET", "edna-project-files-{NAMESPACE}").replace(
    "{NAMESPACE}", K8S_NAMESPACE
)
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
CATEGORICAL_VARIABLES = [
    "site",
    "grtgroup",
    "biome_type",
    "iucn_cat",
    "eco_name",
    "hybas_id",
]
CONTINUOUS_VARIABLES = [
    "bio01",
    "bio12",
    "ghm",
    "elevation",
    "ndvi",
    "average_radiance",
    # All temporal variables (temporal_days, temporal_months, temporal_years) are now categorical
]


def alpha(session: object):
    start_time = time.time()
    ws = ProgressWebSocket("example-report-id")
    ws.send_update("started", 0.0, "Report initialized")

    if not session:
        print("DB session not started. Exiting...")
        exit(1)

    # Example report data - would normally be fetched from database
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
            self.alphaDiversity = 'shannon'
    
    report = MockReport()

    print("Environment configured")

    if report:
        stats_message = ""
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

        # Ensure that the taxonomic rank is in lowercase
        taxonomic_rank = report.taxonomicRank.lower()

        # Handle "max" taxonomic rank (same approach as in beta.py)
        if taxonomic_rank == "max":
            taxonomic_rank = "taxonomic_path"

        # Filter metadata
        metadata = metadata.filter(
            metadata["latitude"].is_not_null()
            & metadata["longitude"].is_not_null()
            & metadata["sample_id"].is_not_null()
        )
        
        # Assuming metadata is a DataFrame
        if len(metadata) == 0:
            exit("Error: Sample data frame is empty. Cannot proceed.")

        # TODO: Convert metadata operations to those of a Polars DataFrame so that we can remove this line.
        metadata = metadata.to_pandas()
        sample = metadata.copy()

        # Set row index to 'sample_id' and remove the 'sample_id' column
        sample.set_index("sample_id", inplace=True)

        # Download the primer file
        update_report_status(session, report.id, ReportBuildState.LOADING.value, False)
        session.commit()

        # Get taxonomic ranks and filter the prevalence
        [taxonomic_ranks, _, taxonomic_num] = get_taxonomic_ranks(
            taxonomic_rank=taxonomic_rank
        )

        # Filter the frequency of taxa per the selected taxonomic rank
        read_results = load_reads_for_primer(
            primer=report.marker,
            project_id=report.project_id,
            taxonomic_ranks=taxonomic_ranks,
            minimum_reads_per_sample=report.countThreshold,
            confidence=report.confidenceLevel,
            minimum_reads_per_taxon=report.filterThreshold,
            metadata=pl.from_pandas(metadata),
            controls=controls["sample_id"].unique().to_list(),
            report_id=report.id,
        )
        tronko_db = read_results["decontaminated_reads"]
        sample_list = read_results["valid_samples"]

        labels_and_legend = get_labels(BUCKET, "analysis/LabelsAndLegends.csv")
        categories = get_labels(BUCKET, "analysis/Categories.csv")

        update_report_status(session, report.id, ReportBuildState.BUILDING.value, False)
        session.commit()

        metadata = adjust_metadata_with_categories(
            metadata,
            categories=categories,
            environmental_variable=report.environmentalParameter,
        )

        metadata = metadata.drop_duplicates(subset="sample_id", keep="first")

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

        tronko_input = tronko_db.to_pandas()

        # Include taxonomic_path in the columns if it's been created
        if (
            taxonomic_rank == "taxonomic_path"
            and "taxonomic_path" in tronko_input.columns
        ):
            tronko_input = tronko_input[
                ["sample_id", "freq", "taxonomic_path"]
                + taxonomic_ranks[0 : taxonomic_num + 1]
            ]
        else:
            tronko_input = tronko_input[
                ["sample_id", "freq"] + taxonomic_ranks[0 : taxonomic_num + 1]
            ]
        print("Filtered by selected Taxonomic Ranks")

        # Remove taxa which are unknown at a given rank
        tronko_input = tronko_input.dropna(subset=[taxonomic_rank])

        # Filter results by species list
        if report.speciesList != "None":
            species_list_names = species_list_df["name"].unique()
            tronko_input = tronko_input[
                tronko_input["sample_id"].isin(metadata["sample_id"].dropna().unique())
                & tronko_input["species"].isin(species_list_names)
            ]
        else:
            tronko_input = tronko_input[
                tronko_input["sample_id"].isin(metadata["sample_id"].dropna().unique())
            ]

        # Create OTU matrix
        otumat = pd.pivot_table(
            tronko_input,
            values="freq",
            index=[taxonomic_rank],
            columns=["sample_id"],
            aggfunc=sum,
            fill_value=0,
        )

        # Convert any object (likely strings) columns to numeric
        otumat = otumat.apply(pd.to_numeric, errors="coerce")

        # Transpose the matrix for skbio
        otumat_transposed = otumat.T
        otumat_transposed

        # Create merged Phyloseq object (skbio doesn't have Phyloseq, but you can use its diversity measures)
        # Just using a single alpha diversity metric 'shannon' as an example
        metric = report.alphaDiversity.lower()
        if metric == "observed":
            metric = "observed_otus"
        ws.send_update("analyzing", 0.85, "Performing alpha diversity calculations")
        alpha_diversity_result = diversity.alpha_diversity(
            metric,
            otumat_transposed.to_numpy(),
            ids=otumat_transposed.index,
        )

        # Convert alpha_diversity to DataFrame
        alpha_df = pd.DataFrame(
            {
                "sample_id": alpha_diversity_result.index,
                "alpha_diversity": alpha_diversity_result.values,
            }
        )

        # Merge metadata with alpha diversity
        merged_df = pd.merge(alpha_df, metadata, how="left", on="sample_id")

        # Initialize empty figure
        fig = None

        # Assume Sample and OTU objects are created here
        ws.send_update("analyzing", 0.9, "Merging environmental data")
        if report.environmentalParameter in CATEGORICAL_VARIABLES:
            # For plotting, let's put data into a new DataFrame
            df = pd.DataFrame(
                {
                    "alpha_diversity": alpha_diversity_result,
                    "EnvironmentalVariable": merged_df.set_index("sample_id")[
                        report.environmentalParameter
                    ],
                }
            )

            # Convert columns to numeric, coerce errors to handle non-numeric data by converting them to NaN
            df["alpha_diversity"] = pd.to_numeric(
                df["alpha_diversity"], errors="coerce"
            )

            # Drop rows where either column is NaN to avoid issues in the correlation test
            df = df.dropna(subset=["EnvironmentalVariable", "alpha_diversity"])

            # Group the data by the environmental variable
            groups = df.groupby("EnvironmentalVariable")["alpha_diversity"].apply(list)

            # Run Kruskal-Wallis test if there's variation in the environmental variable
            if len(df["EnvironmentalVariable"].unique()) > 1:
                h_stat, p_val = stats.kruskal(*groups)
                stats_message = f"chi-squared = {h_stat:.3f}, p-value = {p_val:.3f}"
                print("Ran Kruskal-Wallis test")
            else:
                stats_message = "Not enough variation in environmental variable to run Kruskal-Wallis test."

            # Ensure your DataFrame is sorted by the environmental parameter in alphanumeric order
            merged_df = merged_df.sort_values(by=report.environmentalParameter)

            stats_message = f"{report.alphaDiversity} vs {report.environmentalParameter}\n{stats_message}"

            # Create a violin plot using Plotly
            ws.send_update("analyzing", 0.95, "Plotting results")
            fig = px.violin(
                merged_df,
                x=report.environmentalParameter,
                y="alpha_diversity",
                box=True,
                points="all",
                color=report.environmentalParameter,
                title=stats_message,
                hover_data=[
                    "site",
                    "sample_id",
                ],  # Add site and sample_id to hover data
            )
            fig.update_layout(height=800)  # Set the height to 800 pixels

            # Update hover template to always show site and sample ID
            for i in range(len(fig.data)):
                fig.data[i].hovertemplate = (
                    "<b>Site:</b> %{customdata[0]}<br>"
                    "<b>Sample ID:</b> %{customdata[1]}<br>"
                    "<b>Alpha Diversity:</b> %{y:.4f}<br>"
                    "<extra></extra>"
                )

        if report.environmentalParameter in CONTINUOUS_VARIABLES:
            # Create a DataFrame for plotting
            # Ensure environmental variable is numeric

            merged_df[report.environmentalParameter] = pd.to_numeric(
                merged_df[report.environmentalParameter], errors="coerce"
            )

            # For plotting, let's put data into a new DataFrame
            df = pd.DataFrame(
                {
                    "alpha_diversity": alpha_diversity_result,
                    "EnvironmentalVariable": merged_df.set_index("sample_id")[
                        report.environmentalParameter
                    ],
                }
            )

            # Convert columns to numeric, coerce errors to handle non-numeric data by converting them to NaN
            df["alpha_diversity"] = pd.to_numeric(
                df["alpha_diversity"], errors="coerce"
            )
            df["EnvironmentalVariable"] = pd.to_numeric(
                df["EnvironmentalVariable"], errors="coerce"
            )
            df = df.dropna(subset=["EnvironmentalVariable", "alpha_diversity"])

            # Calculate summary statistics using Kendall correlation test if there's variation
            if len(df["EnvironmentalVariable"].unique()) > 1:
                tau, p_val = stats.kendalltau(
                    df["EnvironmentalVariable"], df["alpha_diversity"]
                )
                stats_message = f"z = {tau:.3f}, tau = {tau:.3f}, p-value = {p_val:.3f}"
            else:
                stats_message = "Not enough variation in environmental variable to run Kendall correlation test."

            # Calculate Lowess curve
            lowess = sm.nonparametric.lowess(
                df["alpha_diversity"], df["EnvironmentalVariable"], frac=0.3
            )  # Adjust 'frac' as needed

            # Extract Lowess curve values
            lowess_x = lowess[:, 0]
            lowess_y = lowess[:, 1]

            # Calculate confidence intervals for Lowess curve
            ci = sm.nonparametric.lowess(
                df["alpha_diversity"],
                df["EnvironmentalVariable"],
                frac=0.3,
                return_sorted=False,
            )

            # Calculate 95% confidence interval
            ci_upper = ci + 1.96 * np.std(df["alpha_diversity"] - ci)
            ci_lower = ci - 1.96 * np.std(df["alpha_diversity"] - ci)

            # Create a scatter plot
            ws.send_update("analyzing", 0.95, "Plotting results")
            fig = go.Figure()

            # Create a merged dataframe that includes site and sample_id for hover data
            scatter_df = df.copy().reset_index()
            scatter_df = pd.merge(
                scatter_df,
                merged_df[["sample_id", "site"]],
                left_on="index",
                right_on="sample_id",
                how="left",
            )

            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=scatter_df["EnvironmentalVariable"],
                    y=scatter_df["alpha_diversity"],
                    mode="markers",
                    name="Data",
                    customdata=scatter_df[["site", "sample_id"]],
                    hovertemplate=(
                        # Skip showing site if the environmental parameter is already site
                        (
                            ""
                            if report.environmentalParameter == "site"
                            else "<b>Site:</b> %{customdata[0]}<br>"
                        )
                        + "<b>Sample ID:</b> %{customdata[1]}<br>"
                        f"<b>{report.alphaDiversity}:</b> %{y:.4f}<br>"
                        f"<b>{report.environmentalParameter}:</b> %{x:.4f}<br>"
                        "<extra></extra>"
                    ),
                )
            ).update_traces(marker=dict(color="black"))

            # Lowess curve
            fig.add_trace(
                go.Scatter(x=lowess_x, y=lowess_y, mode="lines", name="Lowess Curve")
            ).update_traces(line=dict(color="blue"))

            # Confidence interval bands
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([lowess_x, lowess_x[::-1]]),
                    y=np.concatenate([ci_upper, ci_lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(100,100,100,0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="95% Confidence Interval",
                )
            )
            fig.update_layout(showlegend=False)

            stats_message = f"{report.alphaDiversity} vs {report.environmentalParameter}\n{stats_message}"
            # Relabel chart
            fig.update_layout(
                title=stats_message,
                xaxis_title=report.environmentalParameter,
                yaxis_title=report.alphaDiversity,
            )
            # Update chart background
            fig.update_layout(plot_bgcolor="white")
            fig.update_xaxes(
                mirror=True,
                ticks="outside",
                showline=True,
                linecolor="black",
                gridcolor="lightgrey",
            )
            fig.update_yaxes(
                mirror=True,
                ticks="outside",
                showline=True,
                linecolor="black",
                gridcolor="lightgrey",
            )

        # Initialize a dictionary to store sample counts
        sample_db = {"totalSamples": 0, "filteredSamples": 0}

        # Use total and filtered counts from process_metadata
        sample_db["totalSamples"] = total_samples
        sample_db["filteredSamples"] = len(sample_list)
        sample_db["filteredSites"] = filtered_sites
        sample_db["totalSites"] = total_sites

        new_legend, _ = extract_legend_and_axis_label(
            labels_and_legend=labels_and_legend,
            environmental_variable=report.environmentalParameter,
        )

        plotly_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        updated_plotly_json_str = plotly_json.replace(
            f'"{report.environmentalParameter}"', f'"{new_legend}"'
        )
        # Store the results in a list of dictionaries, mimicking the R datasets list
        datasets = {
            "results": [updated_plotly_json_str],
            "metadata": [sample_db],
        }
        file_contents = json.dumps(
            {"datasets": datasets}
        )  # Convert the data to a JSON string

        # Upload report JSON
        object_key = f"projects/{report.project_id}/cache/{report.marker}/reports/{report.id}.json.zst"
        compressed_file = compress_string(file_contents)
        upload_from_bytes(bucket_name=BUCKET, target_path=object_key, bytes=compressed_file)

        # Upload tarball with TSV output
        save_report_tarball(
            [("alpha", pl.from_pandas(merged_df))],
            report,
            BUCKET,
            ["-----------------------------", stats_message],
        )

        # Export plotly visualization as HTML for review
        import os
        os.makedirs("previews", exist_ok=True)
        output_html_path = "previews/alpha_diversity_report.html"
        fig.write_html(output_html_path)
        print(f"Alpha diversity report HTML saved to: {output_html_path}")
        print("📊 Open this file in your browser to view the interactive visualization")

        print(f"Wrote final output to s3://{BUCKET}/{object_key}")

        ws.send_update("completed", 1.0, "Report completed")
        ws.close()

        update_report_status(session, report.id, ReportBuildState.COMPLETED.value, False)
        # Database update removed for mock data challenge
        # session.query(Report).filter(Report.id == report.id).update(
        #     {"fileKey": object_key}, synchronize_session="fetch"
        # )
        compute = ComputeLog(
            project_id=report.project_id,
            description="alpha_diversity_report",
            operation="report",
            executedAt=datetime.fromtimestamp(start_time),
            duration=int((time.time() - start_time) * 1000),
            cores=8,
            memory=8,
        )
        session.add(compute)
        session.commit()
        session.close()

    # Calculate and print the total execution time
    end_time = time.time()  # Capture the end time
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


def main():
    session = start_db_session(K8S_NAMESPACE)

    try:
        print("Starting the alpha script...")
        alpha(session=session)
    except Exception as e:
        print(f"Error in alpha diversity calculation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    main()
