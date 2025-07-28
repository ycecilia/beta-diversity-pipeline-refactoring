"""
Visualization module for beta diversity analysis.
"""

import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import plotly.utils
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from math import cos, sin, pi

from .config import get_config
from .exceptions import VisualizationError
from .logging_config import get_logger, performance_tracker
from .analysis import BetaDiversityResults, PCoAResults


class BetaDiversityVisualizer:
    """Comprehensive visualization for beta diversity analysis."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        self.continuous_variables = [
            "latitude",
            "longitude",
            "temperature",
            "ph",
            "salinity",
            "depth",
            "temporal_months",
            "temporal_days",
            "temporal_years",
        ]

    @performance_tracker("create_pcoa_plot")
    def create_pcoa_plot(
        self,
        results: BetaDiversityResults,
        metadata: pl.DataFrame,
        environmental_param: str,
        title_override: Optional[str] = None,
    ) -> go.Figure:
        """
        Create PCoA visualization plot.

        Args:
            results: Beta diversity analysis results
            metadata: Metadata DataFrame
            environmental_param: Environmental parameter for coloring
            title_override: Optional custom title

        Returns:
            Plotly figure object
        """
        try:
            self.logger.info("Creating PCoA visualization")

            # Prepare data for plotting
            plot_data = self._prepare_plot_data(results, metadata, environmental_param)

            # Determine if environmental parameter is continuous
            is_continuous = environmental_param in self.continuous_variables

            # Create base plot
            if is_continuous:
                fig = self._create_continuous_plot(
                    plot_data, environmental_param, results
                )
            else:
                fig = self._create_categorical_plot(
                    plot_data, environmental_param, results
                )

            # Add title and labels
            title = self._create_plot_title(
                results, environmental_param, title_override
            )

            variance_pc1 = results.metadata.get("variance_explained_pc1", 0)
            variance_pc2 = results.metadata.get("variance_explained_pc2", 0)

            # Get figure size from config with optimization for fast mode
            if hasattr(self.config, "analysis") and self.config.analysis.fast_mode:
                # Smaller figures for faster rendering in fast mode
                width = min(600, self.config.visualization.figure_width)
                height = min(450, self.config.visualization.figure_height)
            else:
                width = self.config.visualization.figure_width
                height = self.config.visualization.figure_height

            fig.update_layout(
                title=title,
                xaxis_title=f"PC1 ({variance_pc1:.2f}% of variance)",
                yaxis_title=f"PC2 ({variance_pc2:.2f}% of variance)",
                width=width,
                height=height,
                margin=dict(l=0, r=0, t=100, b=0, pad=0),
            )

            # Update marker style
            fig.update_traces(
                marker=dict(
                    symbol=self.config.visualization.marker_symbol,
                    size=self.config.visualization.marker_size,
                )
            )

            # Add custom hover templates
            self._update_hover_templates(fig, environmental_param, is_continuous)

            self.logger.info("PCoA plot created successfully")

            return fig

        except Exception as e:
            self.logger.error(f"PCoA plot creation failed: {e}")
            raise VisualizationError(f"PCoA plot creation failed: {e}")

    def _prepare_plot_data(
        self,
        results: BetaDiversityResults,
        metadata: pl.DataFrame,
        environmental_param: str,
    ) -> Dict[str, Any]:
        """Prepare data for plotting."""
        # Get PCoA scores
        pc1_scores = results.sample_scores["PC1"]
        pc2_scores = results.sample_scores["PC2"]
        sample_ids = list(results.distance_matrix.ids)

        # Create PCoA DataFrame
        pcoa_df = pl.DataFrame(
            {"sample_id": sample_ids, "PC1": pc1_scores, "PC2": pc2_scores}
        )

        # Merge with metadata - avoid duplicate columns
        metadata_columns = ["sample_id", "latitude", "longitude"]
        if environmental_param not in metadata_columns:
            metadata_columns.append(environmental_param)
        if "site" not in metadata_columns:
            metadata_columns.append("site")

        metadata_subset = metadata.select(metadata_columns).unique(
            subset=["sample_id"], keep="first"
        )

        plot_df = pcoa_df.join(metadata_subset, on="sample_id", how="left")

        # Handle missing values
        plot_df = plot_df.drop_nulls(subset=[environmental_param])

        return {
            "dataframe": plot_df,
            "plot_data": plot_df.to_dicts(),
            "sample_ids": sample_ids,
            "pc1_range": (
                float(plot_df.select("PC1").min().item()),
                float(plot_df.select("PC1").max().item()),
            ),
            "pc2_range": (
                float(plot_df.select("PC2").min().item()),
                float(plot_df.select("PC2").max().item()),
            ),
        }

    def _create_continuous_plot(
        self,
        plot_data: Dict[str, Any],
        environmental_param: str,
        results: BetaDiversityResults,
    ) -> go.Figure:
        """Create plot for continuous environmental variables."""
        df_dict = plot_data["plot_data"]

        # Ensure environmental parameter is numeric
        for record in df_dict:
            if record.get(environmental_param) is not None:
                try:
                    record[environmental_param] = float(record[environmental_param])
                except (ValueError, TypeError):
                    record[environmental_param] = None

        # Remove records with None values
        df_dict = [r for r in df_dict if r.get(environmental_param) is not None]

        fig = px.scatter(
            df_dict,
            x="PC1",
            y="PC2",
            color=environmental_param,
            hover_data=["site", "sample_id", "latitude", "longitude"],
            color_continuous_scale=self.config.visualization.colorscale_continuous,
        )

        return fig

    def _create_categorical_plot(
        self,
        plot_data: Dict[str, Any],
        environmental_param: str,
        results: BetaDiversityResults,
    ) -> go.Figure:
        """Create plot for categorical environmental variables."""
        df_dict = plot_data["plot_data"]
        df = plot_data["dataframe"]

        # Get unique categories and create color mapping
        categories = sorted(
            df.select(environmental_param).unique().to_series().to_list()
        )
        n_categories = len(categories)

        if n_categories > 1:
            palette = sample_colorscale(
                self.config.visualization.color_palette,
                [i / (n_categories - 1) for i in range(n_categories)],
            )
            color_map = dict(zip(categories, palette))
        else:
            color_map = {categories[0]: "#1f77b4"}

        fig = px.scatter(
            df_dict,
            x="PC1",
            y="PC2",
            color=environmental_param,
            hover_data=["site", "sample_id", "latitude", "longitude"],
            color_discrete_map=color_map,
        )

        return fig

    def _create_plot_title(
        self,
        results: BetaDiversityResults,
        environmental_param: str,
        title_override: Optional[str] = None,
    ) -> str:
        """Create plot title with analysis information."""
        if title_override:
            return title_override

        permanova_info = ""
        if results.permanova_results:
            n_perms = results.permanova_results.get("number of permutations", 999)
            sample_size = results.permanova_results.get("sample size", "N/A")
            n_groups = results.permanova_results.get("number of groups", "N/A")
            f_stat = results.permanova_results.get("test statistic", 0)
            p_value = results.permanova_results.get("p-value", 1)

            permanova_info = (
                f"Results of PERMANOVA, using {n_perms} permutations.<br>"
                f"Sample size: {sample_size}. Number of groups: {n_groups}. "
                f"Test statistic (pseudo-F): {f_stat:.3f}. p-value: {p_value:.3f}"
            )

        metric = results.metadata.get("metric", "braycurtis")
        env_param_display = self._humanize_parameter_name(environmental_param)

        title = f"PCoA plot. {permanova_info}<br>{metric} beta diversity and {env_param_display}"

        return title

    def _humanize_parameter_name(self, param: str) -> str:
        """Convert parameter name to human-readable format."""
        humanized_names = {
            "temporal_months": "Temporal (Months)",
            "temporal_days": "Temporal (Days)",
            "temporal_years": "Temporal (Years)",
            "iucn_cat": "IUCN Category",
            "ph": "pH",
            "site": "Site",
        }

        return humanized_names.get(param, param.replace("_", " ").title())

    def _update_hover_templates(
        self, fig: go.Figure, environmental_param: str, is_continuous: bool
    ) -> None:
        """Update hover templates for better user experience."""
        env_param_display = self._humanize_parameter_name(environmental_param)

        for i, trace in enumerate(fig.data):
            current_name = trace.name if hasattr(trace, "name") else ""

            # Determine site template (skip if environmental parameter is site)
            site_template = (
                ""
                if environmental_param == "site"
                else "<b>Site:</b> %{customdata[0]}<br>"
            )

            # Environmental parameter template
            if is_continuous:
                env_param_template = (
                    f"<b>{env_param_display}:</b> %{{customdata[2]}}<br>"
                )
            else:
                env_param_template = f"<b>{env_param_display}:</b> {current_name}<br>"

            # Update hover template
            trace.hovertemplate = (
                f"{site_template}"
                "<b>Sample ID:</b> %{customdata[1]}<br>"
                f"{env_param_template}"
                "<b>PC1:</b> %{x:.4f}<br>"
                "<b>PC2:</b> %{y:.4f}<br>"
                "<extra></extra>"
            )

    @performance_tracker("add_clustering_ellipses")
    def add_clustering_ellipses(
        self,
        fig: go.Figure,
        pcoa_scores: np.ndarray,
        cluster_labels: np.ndarray,
        colors: List[str],
    ) -> go.Figure:
        """
        Add clustering ellipses to PCoA plot.

        Args:
            fig: Plotly figure object
            pcoa_scores: PCoA coordinate matrix (n_samples x 2)
            cluster_labels: Cluster assignment for each sample
            colors: Colors for each cluster

        Returns:
            Updated figure with ellipses
        """
        try:
            if not self.config.visualization.enable_clustering_ellipses:
                return fig

            self.logger.info("Adding clustering ellipses to plot")

            unique_clusters = np.unique(cluster_labels)

            # Get plot ranges for scaling
            x_range = fig.layout.xaxis.range or [
                pcoa_scores[:, 0].min(),
                pcoa_scores[:, 0].max(),
            ]
            y_range = fig.layout.yaxis.range or [
                pcoa_scores[:, 1].min(),
                pcoa_scores[:, 1].max(),
            ]
            x_range_size = x_range[1] - x_range[0]
            y_range_size = y_range[1] - y_range[0]

            for i, cluster_id in enumerate(unique_clusters):
                if cluster_id == -1:  # Skip noise points in DBSCAN
                    continue

                color = colors[i % len(colors)]
                cluster_points = pcoa_scores[cluster_labels == cluster_id]

                if (
                    len(cluster_points) < 3
                ):  # Need at least 3 points for meaningful ellipse
                    continue

                # Calculate ellipse parameters
                ellipse_params = self._calculate_ellipse_parameters(cluster_points)
                if ellipse_params is None:
                    continue

                center_x, center_y, width, height, angle = ellipse_params

                # Create ellipse path
                ellipse_path = self._create_ellipse_path(
                    center_x, center_y, width, height, angle
                )

                # Add ellipse to plot
                fig.add_shape(
                    type="path",
                    path=ellipse_path,
                    line=dict(color=color, width=3),
                    fillcolor=color,
                    opacity=0.1,
                )

                # Add cluster label
                label_pos = self._calculate_label_position(
                    center_x, center_y, width, height, angle
                )
                radius = min(x_range_size, y_range_size) / 40

                # Add background circle for label
                fig.add_shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=label_pos[0] - radius,
                    y0=label_pos[1] - radius,
                    x1=label_pos[0] + radius,
                    y1=label_pos[1] + radius,
                    fillcolor=color,
                    line_color=color,
                    opacity=1,
                )

                # Add cluster label text
                fig.add_annotation(
                    x=label_pos[0],
                    y=label_pos[1],
                    text=self._int_to_letter(cluster_id + 1),
                    showarrow=False,
                    font=dict(family="Arial", size=14, color="white"),
                    opacity=1,
                    align="center",
                )

            # Ensure equal aspect ratio for ellipses
            fig.update_layout(
                yaxis=dict(scaleanchor="x", scaleratio=1), showlegend=False
            )

            self.logger.info(f"Added ellipses for {len(unique_clusters)} clusters")

            return fig

        except Exception as e:
            self.logger.error(f"Failed to add clustering ellipses: {e}")
            return fig  # Return original figure if ellipses fail

    def _calculate_ellipse_parameters(
        self, points: np.ndarray
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """Calculate ellipse parameters from cluster points."""
        try:
            # Calculate center
            center_x, center_y = np.mean(points, axis=0)

            # Calculate covariance matrix
            cov = np.cov(points.T)

            # Check for valid covariance
            if np.isnan(cov).any() or np.isinf(cov).any():
                return None

            # Calculate eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eig(cov)

            if np.any(eigvals <= 0):
                return None

            # Sort by eigenvalue magnitude
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]

            # Calculate ellipse dimensions (using confidence interval)
            confidence_scale = np.sqrt(
                -2 * np.log(1 - self.config.visualization.ellipse_confidence)
            )
            width = 2 * confidence_scale * np.sqrt(eigvals[0])
            height = 2 * confidence_scale * np.sqrt(eigvals[1])

            # Calculate rotation angle
            angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

            return center_x, center_y, width, height, angle

        except Exception:
            return None

    def _create_ellipse_path(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        angle: float,
    ) -> str:
        """Create SVG path for ellipse."""
        # Generate ellipse points
        t = np.linspace(0, 2 * pi, 100)
        ellipse_x = width / 2 * np.cos(t)
        ellipse_y = height / 2 * np.sin(t)

        # Rotate ellipse
        cos_angle, sin_angle = cos(angle), sin(angle)
        rotated_x = ellipse_x * cos_angle - ellipse_y * sin_angle + center_x
        rotated_y = ellipse_x * sin_angle + ellipse_y * cos_angle + center_y

        # Create path string
        path = f"M {rotated_x[0]},{rotated_y[0]}"
        for i in range(1, len(rotated_x)):
            path += f" L{rotated_x[i]},{rotated_y[i]}"
        path += " Z"

        return path

    def _calculate_label_position(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        angle: float,
    ) -> Tuple[float, float]:
        """Calculate position for cluster label."""
        # Place label at 45 degrees on ellipse
        label_angle = pi / 4

        # Calculate point on ellipse
        a, b = width / 2, height / 2
        cos_label, sin_label = cos(label_angle), sin(label_angle)

        # Ellipse equation: (x/a)^2 + (y/b)^2 = 1
        # Parametric: x = a*cos(t), y = b*sin(t)
        ellipse_x = a * cos_label
        ellipse_y = b * sin_label

        # Rotate point
        cos_angle, sin_angle = cos(angle), sin(angle)
        rotated_x = ellipse_x * cos_angle - ellipse_y * sin_angle + center_x
        rotated_y = ellipse_x * sin_angle + ellipse_y * cos_angle + center_y

        return rotated_x, rotated_y

    def _int_to_letter(self, number: int) -> str:
        """Convert integer to letter (A, B, C, ...)."""
        if number <= 0:
            return "?"
        return chr(ord("A") + number - 1)

    def create_distance_matrix_heatmap(
        self, distance_matrix, title: str = "Beta Diversity Distance Matrix"
    ) -> go.Figure:
        """
        Create heatmap visualization of distance matrix.

        Args:
            distance_matrix: Beta diversity distance matrix
            title: Plot title

        Returns:
            Plotly heatmap figure
        """
        try:
            self.logger.info("Creating distance matrix heatmap")

            # Get sample IDs and distance data
            sample_ids = list(distance_matrix.ids)
            dist_data = distance_matrix.data

            # Create heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=dist_data,
                    x=sample_ids,
                    y=sample_ids,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Distance"),
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title="Samples",
                yaxis_title="Samples",
                width=800,
                height=800,
            )

            return fig

        except Exception as e:
            self.logger.error(f"Distance matrix heatmap creation failed: {e}")
            raise VisualizationError(f"Distance matrix heatmap creation failed: {e}")

    def create_scree_plot(
        self, pcoa_results: PCoAResults, title: str = "PCoA Scree Plot"
    ) -> go.Figure:
        """
        Create scree plot showing variance explained by each PC.

        Args:
            pcoa_results: PCoA analysis results
            title: Plot title

        Returns:
            Plotly scree plot figure
        """
        try:
            self.logger.info("Creating scree plot")

            # Get eigenvalues and calculate proportions
            eigenvalues = pcoa_results.eigenvalues
            proportions = pcoa_results.proportion_explained * 100

            # Limit to first 10 components
            n_components = min(10, len(eigenvalues))
            pc_labels = [f"PC{i+1}" for i in range(n_components)]

            # Create bar plot
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=pc_labels[:n_components],
                        y=proportions[:n_components],
                        marker_color="steelblue",
                    )
                ]
            )

            fig.update_layout(
                title=title,
                xaxis_title="Principal Component",
                yaxis_title="Variance Explained (%)",
                showlegend=False,
            )

            return fig

        except Exception as e:
            self.logger.error(f"Scree plot creation failed: {e}")
            raise VisualizationError(f"Scree plot creation failed: {e}")

    def create_biplot(
        self,
        pcoa_results: PCoAResults,
        loadings: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        title: str = "PCoA Biplot",
    ) -> go.Figure:
        """
        Create biplot with samples and feature loadings.

        Args:
            pcoa_results: PCoA analysis results
            loadings: Feature loadings matrix
            feature_names: Names of features
            title: Plot title

        Returns:
            Plotly biplot figure
        """
        try:
            self.logger.info("Creating biplot")

            # Create scatter plot of samples
            pc1_scores = pcoa_results.sample_scores["PC1"]
            pc2_scores = pcoa_results.sample_scores["PC2"]

            fig = go.Figure()

            # Add sample points
            fig.add_trace(
                go.Scatter(
                    x=pc1_scores,
                    y=pc2_scores,
                    mode="markers",
                    marker=dict(size=8, color="steelblue"),
                    name="Samples",
                )
            )

            # Add feature vectors if provided
            if loadings is not None and feature_names is not None:
                # Scale loadings for visualization
                scale_factor = (
                    max(np.max(np.abs(pc1_scores)), np.max(np.abs(pc2_scores))) * 0.8
                )
                loadings_scaled = loadings * scale_factor

                # Add arrows for significant features
                for i, feature in enumerate(feature_names):
                    if i < loadings_scaled.shape[0]:
                        fig.add_annotation(
                            x=0,
                            y=0,
                            ax=loadings_scaled[i, 0],
                            ay=loadings_scaled[i, 1],
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="red",
                        )

                        fig.add_annotation(
                            x=loadings_scaled[i, 0],
                            y=loadings_scaled[i, 1],
                            text=feature,
                            showarrow=False,
                            font=dict(size=10, color="red"),
                        )

            # Update layout
            variance_pc1 = pcoa_results.proportion_explained[0] * 100
            variance_pc2 = pcoa_results.proportion_explained[1] * 100

            fig.update_layout(
                title=title,
                xaxis_title=f"PC1 ({variance_pc1:.2f}% variance)",
                yaxis_title=f"PC2 ({variance_pc2:.2f}% variance)",
                showlegend=True,
            )

            return fig

        except Exception as e:
            self.logger.error(f"Biplot creation failed: {e}")
            raise VisualizationError(f"Biplot creation failed: {e}")

    def export_plot(
        self,
        fig: go.Figure,
        output_path: Path,
        format: str = "html",
        width: int = 800,
        height: int = 600,
        scale: float = 1.0,
    ) -> None:
        """
        Export plot to file.

        Args:
            fig: Plotly figure
            output_path: Output file path
            format: Export format (html, png, svg, pdf)
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor for image export
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "html":
                fig.write_html(output_path)
            elif format.lower() == "json":
                # Export as JSON for web applications
                plotly_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                with open(output_path, "w") as f:
                    f.write(plotly_json)
            else:
                # For image formats, try to use kaleido (if available)
                try:
                    if format.lower() == "png":
                        fig.write_image(
                            output_path, width=width, height=height, scale=scale
                        )
                    elif format.lower() == "svg":
                        fig.write_image(
                            output_path,
                            format="svg",
                            width=width,
                            height=height,
                            scale=scale,
                        )
                    elif format.lower() == "pd":
                        fig.write_image(
                            output_path,
                            format="pd",
                            width=width,
                            height=height,
                            scale=scale,
                        )
                    else:
                        raise VisualizationError(f"Unsupported export format: {format}")
                except Exception as e:
                    self.logger.warning(
                        f"Image export failed (kaleido not available?): {e}"
                    )
                    # Fallback to HTML
                    html_path = output_path.with_suffix(".html")
                    fig.write_html(html_path)
                    self.logger.info(f"Exported as HTML instead: {html_path}")

            self.logger.info(f"Plot exported to: {output_path}")

        except Exception as e:
            self.logger.error(f"Plot export failed: {e}")
            raise VisualizationError(f"Plot export failed: {e}")

    def create_metadata_summary(self, metadata: pl.DataFrame) -> Dict[str, Any]:
        """
        Create summary of metadata for plot legends and information.

        Args:
            metadata: Metadata DataFrame

        Returns:
            Metadata summary dictionary
        """
        try:
            summary = {
                "total_samples": metadata.height,
                "unique_sites": (
                    metadata.select("site").unique().height
                    if "site" in metadata.columns
                    else 0
                ),
                "coordinate_range": {},
                "environmental_parameters": {},
            }

            # Calculate coordinate ranges
            if "latitude" in metadata.columns:
                summary["coordinate_range"]["latitude"] = {
                    "min": float(metadata.select("latitude").min().item() or 0),
                    "max": float(metadata.select("latitude").max().item() or 0),
                }

            if "longitude" in metadata.columns:
                summary["coordinate_range"]["longitude"] = {
                    "min": float(metadata.select("longitude").min().item() or 0),
                    "max": float(metadata.select("longitude").max().item() or 0),
                }

            # Analyze environmental parameters
            for col in metadata.columns:
                if col not in ["sample_id", "latitude", "longitude", "site"]:
                    if col in self.continuous_variables:
                        # Continuous variable statistics
                        summary["environmental_parameters"][col] = {
                            "type": "continuous",
                            "min": float(metadata.select(col).min().item() or 0),
                            "max": float(metadata.select(col).max().item() or 0),
                            "mean": float(metadata.select(col).mean().item() or 0),
                        }
                    else:
                        # Categorical variable statistics
                        unique_values = (
                            metadata.select(col).unique().to_series().to_list()
                        )
                        summary["environmental_parameters"][col] = {
                            "type": "categorical",
                            "unique_values": len(unique_values),
                            "categories": unique_values[:10],  # Limit to first 10
                        }

            return summary

        except Exception as e:
            self.logger.error(f"Metadata summary creation failed: {e}")
            return {"error": str(e)}
