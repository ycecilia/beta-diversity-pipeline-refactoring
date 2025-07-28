"""
Beta Diversity Analysis Refactored Package

A complete, modular, and high-performance implementation of beta diversity analysis
that replaces the original beta.py script with proper architecture and maintainability.
"""

__version__ = "1.0.0"
__author__ = "Data Analysis Team"

from .pipeline import BetaDiversityPipeline
from .config import Config, get_config
from .exceptions import (
    AnalysisError,
    DataValidationError,
    InsufficientDataError,
    ConfigurationError,
    ProcessingError,
)

__all__ = [
    "BetaDiversityPipeline",
    "Config",
    "get_config",
    "AnalysisError",
    "DataValidationError",
    "InsufficientDataError",
    "ConfigurationError",
    "ProcessingError",
]
