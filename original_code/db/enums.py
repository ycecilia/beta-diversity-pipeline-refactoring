"""
Stub database enums for challenge.
"""
from enum import Enum


class ReportBuildState(Enum):
    """Mock report build states."""
    QUEUED = "queued"
    LOADING = "loading"
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"