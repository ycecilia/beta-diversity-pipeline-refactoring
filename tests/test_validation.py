"""
Unit tests for the validation module.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from beta_diversity_refactored.validation import (
    DataValidator,
    validate_otu_data,
    validate_metadata,
    validate_sample_consistency,
    validate_file_format,
    validate_configuration,
)
from beta_diversity_refactored.exceptions import DataValidationError
from beta_diversity_refactored.config import BetaDiversityConfig


def test_simple_validation():
    """Simple test to check if testing framework works."""
    data = pl.DataFrame(
        {
            "SampleID": ["Sample_001", "Sample_002"],
            "OTU_001": [100, 200],
            "OTU_002": [150, 250],
        }
    )

    result = validate_otu_data(data)
    assert result is True


def test_validation_with_empty_data():
    """Test validation with empty data."""
    empty_data = pl.DataFrame({"SampleID": []})

    with pytest.raises(DataValidationError):
        validate_otu_data(empty_data)


def test_validator_initialization():
    """Test DataValidator initialization."""
    config = BetaDiversityConfig.from_dict({})
    validator = DataValidator(config)

    assert validator.config == config
    assert hasattr(validator, "min_reads_per_sample")
    assert hasattr(validator, "min_prevalence")


def test_validate_otu_data_success():
    """Test successful OTU data validation."""
    data = pl.DataFrame(
        {
            "SampleID": ["Sample_001", "Sample_002", "Sample_003"],
            "OTU_001": [100, 200, 150],
            "OTU_002": [300, 250, 400],
            "OTU_003": [50, 75, 25],
        }
    )

    result = validate_otu_data(data)
    assert result is True


def test_validate_otu_data_missing_sample_id():
    """Test validation with missing SampleID column."""
    invalid_data = pl.DataFrame(
        {"WrongColumn": ["Sample_001", "Sample_002"], "OTU_001": [100, 200]}
    )

    with pytest.raises(DataValidationError, match="SampleID column not found"):
        validate_otu_data(invalid_data)


def test_validate_otu_data_negative_values():
    """Test validation with negative abundance values."""
    invalid_data = pl.DataFrame(
        {
            "SampleID": ["Sample_001", "Sample_002"],
            "OTU_001": [100, -50],
            "OTU_002": [200, 300],
        }
    )

    with pytest.raises(
        DataValidationError,
        match="negative abundance values|Non-numeric abundance values",
    ):
        validate_otu_data(invalid_data)


def test_validate_otu_data_duplicate_samples():
    """Test validation with duplicate sample IDs."""
    invalid_data = pl.DataFrame(
        {
            "SampleID": ["Sample_001", "Sample_001"],
            "OTU_001": [100, 200],
            "OTU_002": [150, 250],
        }
    )

    with pytest.raises(DataValidationError, match="Duplicate sample IDs found"):
        validate_otu_data(invalid_data)


def test_validate_metadata_success():
    """Test successful metadata validation."""
    metadata = pl.DataFrame(
        {
            "SampleID": ["Sample_001", "Sample_002", "Sample_003"],
            "TreatmentGroup": ["Control", "Treatment", "Control"],
        }
    )

    result = validate_metadata(metadata)
    assert result is True


def test_validate_metadata_missing_sample_id():
    """Test validation with missing SampleID column."""
    invalid_metadata = pl.DataFrame(
        {
            "WrongColumn": ["Sample_001", "Sample_002"],
            "TreatmentGroup": ["Control", "Treatment"],
        }
    )

    with pytest.raises(DataValidationError, match="SampleID column not found"):
        validate_metadata(invalid_metadata)


def test_validate_sample_consistency_success():
    """Test successful sample consistency validation."""
    otu_data = pl.DataFrame(
        {
            "SampleID": ["Sample_001", "Sample_002", "Sample_003"],
            "OTU_001": [100, 200, 150],
        }
    )

    metadata = pl.DataFrame(
        {
            "SampleID": ["Sample_001", "Sample_002", "Sample_003"],
            "TreatmentGroup": ["Control", "Treatment", "Control"],
        }
    )

    result = validate_sample_consistency(otu_data, metadata)
    assert result is True


def test_validate_sample_consistency_mismatch():
    """Test validation with completely mismatched samples."""
    otu_data = pl.DataFrame(
        {
            "SampleID": ["Sample_001", "Sample_002", "Sample_003"],
            "OTU_001": [100, 200, 300],
        }
    )

    metadata = pl.DataFrame(
        {
            "SampleID": ["Sample_004", "Sample_005"],
            "TreatmentGroup": ["Control", "Treatment"],
        }
    )

    with pytest.raises(DataValidationError, match="No common samples"):
        validate_sample_consistency(otu_data, metadata)


def test_validate_configuration_success():
    """Test successful configuration validation."""
    config = BetaDiversityConfig.from_dict(
        {
            "analysis": {
                "distance_metric": "braycurtis",
                "n_components": 3,
                "permutations": 999,
            }
        }
    )

    result = validate_configuration(config)
    assert result is True


def test_validate_abundance_data():
    """Test abundance data validation."""
    config = BetaDiversityConfig.from_dict({})
    validator = DataValidator(config)

    # Create data with required columns based on actual expectations
    abundance_data = pl.DataFrame(
        {
            "sample_id": ["Sample_001", "Sample_002", "Sample_003"],
            "OTU_001": ["OTU_A", "OTU_B", "OTU_A"],
            "reads": [100, 200, 150],
        }
    )

    # Test with OTU_001 as taxonomic rank
    result = validator.validate_abundance_data(abundance_data, "OTU_001")
    assert isinstance(result, dict)
