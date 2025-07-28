"""
Configuration management for beta diversity analysis.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml
import json

from .exceptions import ConfigurationError


@dataclass
class DatabaseConfig:
    """Database configuration."""

    namespace: str = "staging"
    connection_string: Optional[str] = None
    session_timeout: int = 30


@dataclass
class StorageConfig:
    """Storage configuration."""

    bucket: str = "edna-project-files-staging"
    cache_dir: str = "./cache"
    output_dir: str = "./output"
    temp_dir: str = "./temp"


@dataclass
class AnalysisConfig:
    """Analysis configuration."""

    # Beta diversity metrics
    default_metric: str = "braycurtis"
    supported_metrics: List[str] = field(
        default_factory=lambda: [
            "braycurtis",
            "jaccard",
            "dice",
            "hamming",
            "cityblock",
            "cosine",
            "euclidean",
            "manhattan",
            "chebyshev",
        ]
    )

    # PERMANOVA settings
    permanova_permutations: int = 999
    permanova_method: str = "centroid"

    # PCoA settings
    pcoa_method: str = "fsvd"
    pcoa_inplace: bool = True

    # Clustering settings
    clustering_method: str = "meanshift"
    supported_clustering: List[str] = field(
        default_factory=lambda: ["meanshift", "optics", "balanced", "kmeans", "dbscan"]
    )

    # Performance settings
    enable_streaming: bool = False
    chunk_size: int = 10000
    parallel_workers: int = 4
    memory_limit_mb: int = 8192
    fast_mode: bool = False  # Skip non-essential validations for speed
    enable_multiprocessing: bool = True  # Enable parallel processing where possible

    # Advanced performance optimizations
    enable_lazy_loading: bool = True  # Use lazy DataFrames where possible
    optimize_memory: bool = True  # Enable memory optimizations
    enable_caching: bool = True  # Enable intermediate result caching
    cache_distance_matrices: bool = True  # Cache computed distance matrices
    use_float32: bool = True  # Use float32 instead of float64 for memory savings
    parallel_clustering: bool = True  # Parallelize clustering operations
    max_pcoa_dimensions: int = 10  # Limit PCoA dimensions for performance
    enable_compression: bool = True  # Compress intermediate data
    optimize_io: bool = True  # Optimize file I/O operations


@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    color_palette: str = "Turbo"
    colorscale_continuous: str = "Viridis"
    marker_symbol: str = "square"
    marker_size: int = 8
    figure_width: int = 800
    figure_height: int = 600
    enable_clustering_ellipses: bool = True
    ellipse_confidence: float = 0.95


@dataclass
class ValidationConfig:
    """Data validation configuration."""

    min_samples: int = 3
    min_taxa: int = 2
    min_reads_per_sample: int = 100
    min_reads_per_taxon: int = 10
    max_missing_percentage: float = 0.5
    coordinate_precision: int = 6
    fast_mode: bool = False  # Skip detailed validations for speed


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    enable_performance_logging: bool = True
    enable_memory_tracking: bool = True


@dataclass
class Config:
    """Main configuration class."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._apply_environment_overrides()

    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate analysis config
        if self.analysis.default_metric not in self.analysis.supported_metrics:
            raise ConfigurationError(
                f"Default metric '{self.analysis.default_metric}' not in "
                "supported metrics"
            )

        if self.analysis.clustering_method not in self.analysis.supported_clustering:
            raise ConfigurationError(
                f"Clustering method '{self.analysis.clustering_method}' not "
                "supported"
            )

        # Validate validation config
        if self.validation.min_samples < 2:
            raise ConfigurationError("Minimum samples must be at least 2")

        if (
            self.validation.max_missing_percentage < 0
            or self.validation.max_missing_percentage > 1
        ):
            raise ConfigurationError("Max missing percentage must be between 0 and 1")

        # Validate visualization config
        if (
            self.visualization.ellipse_confidence <= 0
            or self.visualization.ellipse_confidence >= 1
        ):
            raise ConfigurationError("Ellipse confidence must be between 0 and 1")

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Database overrides
        if namespace := os.getenv("K8S_NAMESPACE"):
            self.database.namespace = namespace

        # Storage overrides
        if bucket := os.getenv("GCS_BUCKET"):
            self.storage.bucket = bucket.replace("{NAMESPACE}", self.database.namespace)

        if cache_dir := os.getenv("CACHE_DIR"):
            self.storage.cache_dir = cache_dir

        # Analysis overrides
        if metric := os.getenv("BETA_DIVERSITY_METRIC"):
            if metric in self.analysis.supported_metrics:
                self.analysis.default_metric = metric

        if clustering := os.getenv("CLUSTERING_METHOD"):
            if clustering in self.analysis.supported_clustering:
                self.analysis.clustering_method = clustering

        # Performance overrides
        if workers := os.getenv("PARALLEL_WORKERS"):
            try:
                self.analysis.parallel_workers = int(workers)
            except ValueError:
                pass

        if memory_limit := os.getenv("MEMORY_LIMIT_MB"):
            try:
                self.analysis.memory_limit_mb = int(memory_limit)
            except ValueError:
                pass

        # Logging overrides
        if log_level := os.getenv("LOG_LEVEL"):
            self.logging.level = log_level.upper()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "database": self.database.__dict__,
            "storage": self.storage.__dict__,
            "analysis": self.analysis.__dict__,
            "visualization": self.visualization.__dict__,
            "validation": self.validation.__dict__,
            "logging": self.logging.__dict__,
        }

    def save_to_file(self, file_path: Path) -> None:
        """Save configuration to file."""
        config_dict = self.to_dict()

        if file_path.suffix.lower() == ".yaml" or file_path.suffix.lower() == ".yml":
            with open(file_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif file_path.suffix.lower() == ".json":
            with open(file_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ConfigurationError(
                f"Unsupported config file format: {file_path.suffix}"
            )

    @classmethod
    def from_file(cls, file_path: Path) -> "Config":
        """Load configuration from file."""
        if not file_path.exists():
            raise ConfigurationError(f"Config file not found: {file_path}")

        try:
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                with open(file_path, "r") as f:
                    config_dict = yaml.safe_load(f)
            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r") as f:
                    config_dict = json.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported config file format: {file_path.suffix}"
                )

            # Create nested config objects
            database_config = DatabaseConfig(**config_dict.get("database", {}))
            storage_config = StorageConfig(**config_dict.get("storage", {}))
            analysis_config = AnalysisConfig(**config_dict.get("analysis", {}))
            viz_config = VisualizationConfig(**config_dict.get("visualization", {}))
            validation_config = ValidationConfig(**config_dict.get("validation", {}))
            logging_config = LoggingConfig(**config_dict.get("logging", {}))

            return cls(
                database=database_config,
                storage=storage_config,
                analysis=analysis_config,
                visualization=viz_config,
                validation=validation_config,
                logging=logging_config,
            )

        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {file_path}: {e}")


# New configuration structure for beta diversity analysis
@dataclass
class InputConfig:
    """Input file configuration."""

    otu_data_file: Optional[str] = None
    metadata_file: Optional[str] = None
    controls_file: Optional[str] = None
    decontaminated_file: Optional[str] = None
    batch_files: Optional[List[Dict[str, str]]] = None
    file_format: str = "csv"


@dataclass
class ProcessingConfig:
    """Data processing configuration."""

    min_reads_per_sample: int = 1000
    min_prevalence: float = 0.01
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    abundance_threshold: float = 0.0001
    normalization_method: str = "relative"
    enable_streaming: bool = False
    chunk_size: int = 10000
    use_parallel: bool = True
    n_jobs: int = -1
    resume_from_checkpoint: bool = False
    enable_batch_processing: bool = False


@dataclass
class BetaAnalysisConfig:
    """Beta diversity analysis configuration."""

    distance_metric: str = "braycurtis"
    supported_metrics: List[str] = field(
        default_factory=lambda: [
            "braycurtis",
            "jaccard",
            "dice",
            "hamming",
            "cityblock",
            "cosine",
            "euclidean",
            "manhattan",
            "chebyshev",
        ]
    )
    supported_clustering: List[str] = field(
        default_factory=lambda: ["meanshift", "optics", "balanced", "kmeans", "dbscan"]
    )
    pcoa_method: str = "fsvd"
    n_components: int = 3
    permutations: int = 999
    enable_permanova: bool = True
    random_state: int = 42


@dataclass
class ClusteringConfig:
    """Clustering configuration."""

    methods: List[str] = field(default_factory=lambda: ["kmeans"])
    n_clusters: int = 3
    linkage_method: str = "ward"
    distance_threshold: Optional[float] = None


@dataclass
class BetaVisualizationConfig:
    """Visualization configuration."""

    create_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300
    figure_size: tuple = (10, 8)
    color_palette: str = "viridis"
    marker_symbol: str = "circle"
    marker_size: int = 8
    colorscale_continuous: str = "viridis"
    enable_clustering_ellipses: bool = True
    ellipse_confidence: float = 0.95


@dataclass
class OutputConfig:
    """Output configuration."""

    results_dir: str = "./results"
    plots_dir: str = "./plots"
    data_dir: str = "./data"
    enable_compression: bool = False
    save_intermediate: bool = False


@dataclass
class BetaLoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    enable_performance_monitoring: bool = False


@dataclass
class BetaDiversityConfig:
    """Main beta diversity configuration class."""

    input: InputConfig = field(default_factory=InputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    analysis: BetaAnalysisConfig = field(default_factory=BetaAnalysisConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    visualization: BetaVisualizationConfig = field(
        default_factory=BetaVisualizationConfig
    )
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: BetaLoggingConfig = field(default_factory=BetaLoggingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BetaDiversityConfig":
        """Create configuration from dictionary."""
        # Extract nested configurations
        input_config = InputConfig(**config_dict.get("input", {}))
        processing_config = ProcessingConfig(**config_dict.get("processing", {}))
        analysis_config = BetaAnalysisConfig(**config_dict.get("analysis", {}))
        clustering_config = ClusteringConfig(**config_dict.get("clustering", {}))
        visualization_config = BetaVisualizationConfig(
            **config_dict.get("visualization", {})
        )
        output_config = OutputConfig(**config_dict.get("output", {}))
        logging_config = BetaLoggingConfig(**config_dict.get("logging", {}))
        validation_config = ValidationConfig(**config_dict.get("validation", {}))

        return cls(
            input=input_config,
            processing=processing_config,
            analysis=analysis_config,
            clustering=clustering_config,
            visualization=visualization_config,
            output=output_config,
            logging=logging_config,
            validation=validation_config,
        )

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "BetaDiversityConfig":
        """Load configuration from file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise ConfigurationError(f"Config file not found: {file_path}")

        try:
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                with open(file_path, "r") as f:
                    config_dict = yaml.safe_load(f)
            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r") as f:
                    config_dict = json.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported config file format: {file_path.suffix}"
                )

            return cls.from_dict(config_dict)

        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {file_path}: {e}")


# Global configuration instance
_config: Optional[Config] = None


def get_config(
    config_path: Optional[Path] = None, force_reload: bool = False
) -> Config:
    """
    Get the global configuration instance.

    Args:
        config_path: Optional path to configuration file
        force_reload: Whether to force reload the configuration

    Returns:
        Configuration instance
    """
    global _config

    if _config is None or force_reload:
        if config_path and config_path.exists():
            _config = Config.from_file(config_path)
        else:
            _config = Config()

    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
