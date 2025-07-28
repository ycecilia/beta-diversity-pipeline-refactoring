"""
Custom exceptions for beta diversity analysis.
"""


class BetaDiversityError(Exception):
    """Base exception for beta diversity analysis."""

    pass


class AnalysisError(BetaDiversityError):
    """Raised when analysis calculations fail."""

    pass


class DataValidationError(BetaDiversityError):
    """Raised when data validation fails."""

    pass


class InsufficientDataError(BetaDiversityError):
    """Raised when there's insufficient data for analysis."""

    pass


class ConfigurationError(BetaDiversityError):
    """Raised when configuration is invalid."""

    pass


class ProcessingError(BetaDiversityError):
    """Raised when data processing fails."""

    pass


class VisualizationError(BetaDiversityError):
    """Raised when visualization creation fails."""

    pass


class StorageError(BetaDiversityError):
    """Raised when storage operations fail."""

    pass


class ClusteringError(BetaDiversityError):
    """Raised when clustering analysis fails."""

    pass


class DataProcessingError(BetaDiversityError):
    """Raised when data processing operations fail."""

    pass


class PipelineError(BetaDiversityError):
    """Raised when pipeline execution fails."""

    pass
