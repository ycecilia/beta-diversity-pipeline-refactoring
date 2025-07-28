"""
Storage and I/O module for beta diversity analysis.
"""

import json
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import polars as pl
import numpy as np
import plotly.utils

from .config import get_config
from .exceptions import StorageError
from .logging_config import get_logger, performance_tracker
from .analysis import BetaDiversityResults


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays and other numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if obj != obj:  # NaN check
            return None
        return super().default(obj)


class StorageManager:
    """Comprehensive storage management for beta diversity analysis."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)

        # Ensure output directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        # Handle both config types
        if hasattr(self.config, "storage"):
            # Original Config class
            directories = [
                self.config.storage.output_dir,
                self.config.storage.cache_dir,
                self.config.storage.temp_dir,
            ]
        else:
            # BetaDiversityConfig class
            directories = [
                self.config.output.results_dir,
                getattr(self.config.output, "cache_dir", "./cache"),
                getattr(self.config.output, "temp_dir", "./temp"),
            ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @performance_tracker("save_results")
    def save_results(
        self,
        results: BetaDiversityResults,
        output_prefix: str,
        include_distance_matrix: bool = True,
        include_plots: bool = True,
        compress: bool = True,
    ) -> Dict[str, Path]:
        """
        Save complete beta diversity results.

        Args:
            results: Beta diversity analysis results
            output_prefix: Prefix for output files
            include_distance_matrix: Whether to save distance matrix
            include_plots: Whether to save plot data
            compress: Whether to compress large files

        Returns:
            Dictionary of saved file paths
        """
        try:
            self.logger.info(f"Saving beta diversity results: {output_prefix}")

            output_dir = Path(self._get_output_dir())
            saved_files = {}

            # Save metadata summary
            metadata_file = output_dir / f"{output_prefix}_metadata.json"
            self._save_json(results.metadata, metadata_file)
            saved_files["metadata"] = metadata_file

            # Save PCoA results
            pcoa_file = output_dir / f"{output_prefix}_pcoa.json"
            pcoa_data = {
                "eigenvalues": results.pcoa_results.eigenvalues.tolist(),
                "proportion_explained": results.pcoa_results.proportion_explained.tolist(),
                "sample_scores": {
                    k: v.tolist() for k, v in results.sample_scores.items()
                },
                "total_variance": float(results.pcoa_results.total_variance),
            }
            self._save_json(pcoa_data, pcoa_file)
            saved_files["pcoa"] = pcoa_file

            # Save PERMANOVA results if available
            if results.permanova_results:
                permanova_file = output_dir / f"{output_prefix}_permanova.json"
                self._save_json(results.permanova_results, permanova_file)
                saved_files["permanova"] = permanova_file

            # Save distance matrix if requested
            if include_distance_matrix:
                dm_file = output_dir / f"{output_prefix}_distance_matrix"
                if compress:
                    dm_file = dm_file.with_suffix(".pkl.gz")
                    self._save_distance_matrix_compressed(
                        results.distance_matrix, dm_file
                    )
                else:
                    dm_file = dm_file.with_suffix(".csv")
                    self._save_distance_matrix_csv(results.distance_matrix, dm_file)
                saved_files["distance_matrix"] = dm_file

            # Save complete results as pickle for later loading
            results_file = output_dir / f"{output_prefix}_complete_results.pkl"
            if compress:
                results_file = results_file.with_suffix(".pkl.gz")
            self._save_pickle(results, results_file, compress=compress)
            saved_files["complete_results"] = results_file

            self.logger.info(f"Results saved: {len(saved_files)} files")

            return saved_files

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise StorageError(f"Failed to save results: {e}")

    @performance_tracker("save_data_frames")
    def save_data_frames(
        self,
        data_frames: Dict[str, pl.DataFrame],
        output_prefix: str,
        format: str = "csv",
    ) -> Dict[str, Path]:
        """
        Save multiple DataFrames.

        Args:
            data_frames: Dictionary of name -> DataFrame
            output_prefix: Prefix for output files
            format: Output format (csv, tsv, parquet, json)

        Returns:
            Dictionary of saved file paths
        """
        try:
            self.logger.info(f"Saving {len(data_frames)} DataFrames: {output_prefix}")

            output_dir = Path(self._get_output_dir())
            saved_files = {}

            for name, df in data_frames.items():
                filename = f"{output_prefix}_{name}.{format}"
                file_path = output_dir / filename

                if format.lower() == "csv":
                    df.write_csv(file_path)
                elif format.lower() == "tsv":
                    df.write_csv(file_path, separator="\t")
                elif format.lower() == "parquet":
                    df.write_parquet(file_path)
                elif format.lower() == "json":
                    df.write_json(file_path)
                else:
                    raise StorageError(f"Unsupported format: {format}")

                saved_files[name] = file_path

            self.logger.info(f"DataFrames saved: {len(saved_files)} files")

            return saved_files

        except Exception as e:
            self.logger.error(f"Failed to save DataFrames: {e}")
            raise StorageError(f"Failed to save DataFrames: {e}")

    def save_plot_data(
        self,
        figure_data: Dict[str, Any],
        output_prefix: str,
        include_html: bool = True,
        include_json: bool = True,
    ) -> Dict[str, Path]:
        """
        Save plot data and figures.

        Args:
            figure_data: Dictionary containing figure and metadata
            output_prefix: Prefix for output files
            include_html: Whether to save HTML version
            include_json: Whether to save JSON version

        Returns:
            Dictionary of saved file paths
        """
        try:
            self.logger.info(f"Saving plot data: {output_prefix}")

            output_dir = Path(self._get_output_dir())
            saved_files = {}

            # Save HTML version if requested
            if include_html and "figure" in figure_data:
                html_file = output_dir / f"{output_prefix}_plot.html"
                figure_data["figure"].write_html(html_file)
                saved_files["html"] = html_file

            # Save JSON version if requested
            if include_json and "figure" in figure_data:
                json_file = output_dir / f"{output_prefix}_plot.json"
                plotly_json = json.dumps(
                    figure_data["figure"], cls=plotly.utils.PlotlyJSONEncoder
                )
                with open(json_file, "w") as f:
                    f.write(plotly_json)
                saved_files["json"] = json_file

            # Save plot metadata
            if "metadata" in figure_data:
                metadata_file = output_dir / f"{output_prefix}_plot_metadata.json"
                self._save_json(figure_data["metadata"], metadata_file)
                saved_files["metadata"] = metadata_file

            self.logger.info(f"Plot data saved: {len(saved_files)} files")

            return saved_files

        except Exception as e:
            self.logger.error(f"Failed to save plot data: {e}")
            raise StorageError(f"Failed to save plot data: {e}")

    def save_analysis_report(
        self,
        results: BetaDiversityResults,
        metadata: Dict[str, Any],
        output_prefix: str,
    ) -> Path:
        """
        Save comprehensive analysis report.

        Args:
            results: Beta diversity analysis results
            metadata: Additional metadata
            output_prefix: Prefix for output file

        Returns:
            Path to saved report
        """
        try:
            self.logger.info(f"Saving analysis report: {output_prefix}")

            output_dir = Path(self._get_output_dir())
            report_file = output_dir / f"{output_prefix}_report.json"

            # Compile comprehensive report
            report = {
                "timestamp": datetime.now().isoformat(),
                "analysis_metadata": results.metadata,
                "additional_metadata": metadata,
                "pcoa_summary": {
                    "variance_explained_pc1": float(
                        results.pcoa_results.proportion_explained[0] * 100
                    ),
                    "variance_explained_pc2": float(
                        results.pcoa_results.proportion_explained[1] * 100
                    ),
                    "total_variance": float(results.pcoa_results.total_variance),
                    "n_components": len(results.pcoa_results.eigenvalues),
                },
                "permanova_summary": results.permanova_results,
                "sample_summary": {
                    "n_samples": len(results.distance_matrix.ids),
                    "sample_ids": list(results.distance_matrix.ids),
                },
            }

            # Add configuration information
            try:
                from dataclasses import asdict

                report["configuration"] = asdict(self.config)
            except Exception:
                # Fallback if asdict doesn't work
                report["configuration"] = {
                    "type": type(self.config).__name__,
                    "description": "Configuration object (serialization not available)",
                }

            self._save_json(report, report_file)

            self.logger.info(f"Analysis report saved: {report_file}")

            return report_file

        except Exception as e:
            self.logger.error(f"Failed to save analysis report: {e}")
            raise StorageError(f"Failed to save analysis report: {e}")

    @performance_tracker("load_results")
    def load_results(self, results_file: Path) -> BetaDiversityResults:
        """
        Load previously saved beta diversity results.

        Args:
            results_file: Path to results file

        Returns:
            Loaded beta diversity results
        """
        try:
            self.logger.info(f"Loading results from: {results_file}")

            if not results_file.exists():
                raise StorageError(f"Results file not found: {results_file}")

            # Determine if file is compressed
            compress = results_file.suffix == ".gz"

            results = self._load_pickle(results_file, compress=compress)

            self.logger.info("Results loaded successfully")

            return results

        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            raise StorageError(f"Failed to load results: {e}")

    def load_data_frame(self, file_path: Path) -> pl.DataFrame:
        """
        Load DataFrame from file with automatic format detection.

        Args:
            file_path: Path to data file

        Returns:
            Loaded DataFrame
        """
        try:
            if not file_path.exists():
                raise StorageError(f"File not found: {file_path}")

            suffix = file_path.suffix.lower()

            if suffix == ".csv":
                return pl.read_csv(file_path)
            elif suffix in [".tsv", ".txt"]:
                return pl.read_csv(file_path, separator="\t")
            elif suffix == ".parquet":
                return pl.read_parquet(file_path)
            elif suffix == ".json":
                return pl.read_json(file_path)
            else:
                # Try CSV as default
                try:
                    return pl.read_csv(file_path)
                except Exception:
                    # Try TSV as fallback
                    return pl.read_csv(file_path, separator="\t")

        except Exception as e:
            self.logger.error(f"Failed to load DataFrame: {e}")
            raise StorageError(f"Failed to load DataFrame from {file_path}: {e}")

    def create_tarball(
        self, files: List[Path], output_path: Path, include_metadata: bool = True
    ) -> Path:
        """
        Create tarball of analysis results.

        Args:
            files: List of files to include
            output_path: Output tarball path
            include_metadata: Whether to include metadata file

        Returns:
            Path to created tarball
        """
        try:
            import tarfile

            self.logger.info(f"Creating tarball: {output_path}")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with tarfile.open(output_path, "w:gz") as tar:
                for file_path in files:
                    if file_path.exists():
                        tar.add(file_path, arcname=file_path.name)

                # Add metadata file if requested
                if include_metadata:
                    metadata = {
                        "created": datetime.now().isoformat(),
                        "files": [f.name for f in files if f.exists()],
                        "total_files": len([f for f in files if f.exists()]),
                    }

                    # Create temporary metadata file
                    temp_metadata = Path(self._get_temp_dir()) / "tarball_metadata.json"
                    self._save_json(metadata, temp_metadata)
                    tar.add(temp_metadata, arcname="metadata.json")
                    temp_metadata.unlink()  # Clean up

            self.logger.info(f"Tarball created: {output_path}")

            return output_path

        except Exception as e:
            self.logger.error(f"Failed to create tarball: {e}")
            raise StorageError(f"Failed to create tarball: {e}")

    def _save_json(self, data: Any, file_path: Path) -> None:
        """Save data as JSON file."""
        with open(file_path, "w") as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)

    def _load_json(self, file_path: Path) -> Any:
        """Load data from JSON file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def _save_pickle(self, data: Any, file_path: Path, compress: bool = False) -> None:
        """Save data as pickle file."""
        if compress:
            with gzip.open(file_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(file_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, file_path: Path, compress: bool = False) -> Any:
        """Load data from pickle file."""
        if compress:
            with gzip.open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            with open(file_path, "rb") as f:
                return pickle.load(f)

    def _save_distance_matrix_csv(self, distance_matrix, file_path: Path) -> None:
        """Save distance matrix as CSV file."""
        # Convert to DataFrame for easier CSV export
        sample_ids = list(distance_matrix.ids)
        df = pl.DataFrame(distance_matrix.data, schema=sample_ids)
        df = df.with_columns(pl.Series("sample_id", sample_ids)).select(
            ["sample_id"] + sample_ids
        )
        df.write_csv(file_path)

    def _save_distance_matrix_compressed(
        self, distance_matrix, file_path: Path
    ) -> None:
        """Save distance matrix as compressed pickle."""
        dm_data = {"data": distance_matrix.data, "ids": list(distance_matrix.ids)}
        self._save_pickle(dm_data, file_path, compress=True)

    def get_cache_path(self, cache_key: str) -> Path:
        """Get path for cached data."""
        cache_dir = Path(self._get_cache_dir())
        return cache_dir / f"{cache_key}.pkl.gz"

    def save_to_cache(self, data: Any, cache_key: str) -> Path:
        """Save data to cache."""
        try:
            cache_path = self.get_cache_path(cache_key)
            self._save_pickle(data, cache_path, compress=True)
            self.logger.debug(f"Data cached: {cache_key}")
            return cache_path
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")
            raise StorageError(f"Failed to cache data: {e}")

    def load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache."""
        try:
            cache_path = self.get_cache_path(cache_key)
            if cache_path.exists():
                data = self._load_pickle(cache_path, compress=True)
                self.logger.debug(f"Data loaded from cache: {cache_key}")
                return data
            return None
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return None

    def clear_cache(self, pattern: str = "*") -> int:
        """
        Clear cached data.

        Args:
            pattern: Glob pattern for files to clear

        Returns:
            Number of files removed
        """
        try:
            cache_dir = Path(self._get_cache_dir())
            files_removed = 0

            for cache_file in cache_dir.glob(pattern):
                if cache_file.is_file():
                    cache_file.unlink()
                    files_removed += 1

            self.logger.info(f"Cache cleared: {files_removed} files removed")

            return files_removed

        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            raise StorageError(f"Failed to clear cache: {e}")

    def get_storage_summary(self) -> Dict[str, Any]:
        """Get summary of storage usage."""
        try:
            summary = {"directories": {}, "total_size": 0}

            directories = [
                ("output", self._get_output_dir()),
                ("cache", self._get_cache_dir()),
                ("temp", self._get_temp_dir()),
            ]

            for name, directory in directories:
                dir_path = Path(directory)
                if dir_path.exists():
                    files = list(dir_path.rglob("*"))
                    file_count = len([f for f in files if f.is_file()])
                    dir_size = sum(f.stat().st_size for f in files if f.is_file())

                    summary["directories"][name] = {
                        "path": str(dir_path),
                        "file_count": file_count,
                        "size_bytes": dir_size,
                        "size_mb": dir_size / (1024 * 1024),
                    }

                    summary["total_size"] += dir_size
                else:
                    summary["directories"][name] = {
                        "path": str(dir_path),
                        "file_count": 0,
                        "size_bytes": 0,
                        "size_mb": 0,
                    }

            summary["total_size_mb"] = summary["total_size"] / (1024 * 1024)

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get storage summary: {e}")
            return {"error": str(e)}

    def _get_output_dir(self) -> str:
        """Get output directory from config, handling both config types."""
        if hasattr(self.config, "storage"):
            return self.config.storage.output_dir
        else:
            return self.config.output.results_dir

    def _get_cache_dir(self) -> str:
        """Get cache directory from config, handling both config types."""
        if hasattr(self.config, "storage"):
            return self.config.storage.cache_dir
        else:
            return getattr(self.config.output, "cache_dir", "./cache")

    def _get_temp_dir(self) -> str:
        """Get temp directory from config, handling both config types."""
        if hasattr(self.config, "storage"):
            return self.config.storage.temp_dir
        else:
            return getattr(self.config.output, "temp_dir", "./temp")
