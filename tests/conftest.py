"""
Shared test fixtures for beta diversity tests.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def sample_otu_data():
    """Sample OTU data for testing."""
    return pl.DataFrame(
        {
            "SampleID": ["S1", "S2", "S3", "S4"],
            "OTU_1": [100, 150, 200, 120],
            "OTU_2": [80, 90, 110, 95],
            "OTU_3": [50, 60, 40, 55],
        }
    )


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return pl.DataFrame(
        {
            "SampleID": ["S1", "S2", "S3", "S4"],
            "site": ["A", "A", "B", "B"],
            "sample_id": ["S1", "S2", "S3", "S4"],
            "latitude": [40.0, 40.1, 41.0, 41.1],
            "longitude": [-74.0, -74.1, -75.0, -75.1],
        }
    )


@pytest.fixture
def temp_dir():
    """Temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)
