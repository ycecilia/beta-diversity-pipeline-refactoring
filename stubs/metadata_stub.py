"""
Stubbed implementation of process_metadata function for testing.

This replaces the original database-dependent process_metadata function
with a local CSV-based implementation for development and testing.
"""
import os
import polars as pl
from typing import Dict, Any, Optional, List
from pathlib import Path


def process_metadata(
    session: Optional[object] = None,
    project_id: str = "proj_123",
    filter_site_ids: Optional[List[str]] = None,
    filter_tag_ids: Optional[List[str]] = None,
    sample_first_date: Optional[str] = None,
    sample_last_date: Optional[str] = None,
    environmental_variable: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stubbed version of process_metadata that loads data from local CSV files.
    
    This function replaces the original database-dependent process_metadata
    to allow developers to test their refactored code without needing the
    full database infrastructure.
    
    Note: Some filtering has been removed for simplicity. The signature is the 
    same as the original function.

    Args:
        session: Database session (ignored in stub)
        project_id: Project ID to filter by
        filter_site_ids: List of site IDs to include
        filter_tag_ids: List of tag IDs to include (not used in stub)
        sample_first_date: Start date for filtering samples
        sample_last_date: End date for filtering samples
        environmental_variable: Environmental variable of interest
        
    Returns:
        Dictionary containing:
        - metadata: Polars DataFrame with sample metadata
        - controls: Polars DataFrame with control samples
        - total_samples: Total number of samples
        - total_sites: Total number of sites
        - filtered_sites: Number of sites after filtering
    """
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    test_data_dir = script_dir.parent / "test_data"
    
    # Load metadata from CSV
    metadata_path = test_data_dir / "sample_metadata.csv"
    controls_path = test_data_dir / "sample_controls.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Test metadata file not found: {metadata_path}")
    if not controls_path.exists():
        raise FileNotFoundError(f"Test controls file not found: {controls_path}")
    
    # Load the data
    metadata = pl.read_csv(metadata_path)
    controls = pl.read_csv(controls_path)
    
    # Apply site filters if provided
    if filter_site_ids:
        metadata = metadata.filter(pl.col("site").is_in(filter_site_ids))
    
    # Calculate summary statistics
    total_samples = len(metadata)
    total_sites = metadata.select("site").n_unique()
    filtered_sites = metadata.select("site").n_unique()  # Same as total in this stub
    
    # Ensure the environmental variable exists if specified
    if environmental_variable and environmental_variable not in metadata.columns:
        print(f"Warning: Environmental variable '{environmental_variable}' not found in metadata")
        print(f"Available columns: {metadata.columns}")
    
    return {
        "metadata": metadata,
        "controls": controls,
        "total_samples": total_samples,
        "total_sites": total_sites,
        "filtered_sites": filtered_sites,
    }


def load_reads_for_primer(
    primer: str = "test_primer",
    project_id: str = "proj_123",
    taxonomic_ranks: Optional[List[str]] = None,
    minimum_reads_per_sample: int = 100,
    confidence: float = 0.8,
    minimum_reads_per_taxon: int = 10,
    metadata: Optional[pl.DataFrame] = None,
    controls: Optional[List[str]] = None,
    report_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stubbed version of load_reads_for_primer that loads OTU data from CSV.
    
    Args:
        primer: Primer name (ignored in stub)
        project_id: Project ID to filter by
        taxonomic_ranks: List of taxonomic ranks to include
        minimum_reads_per_sample: Minimum reads threshold per sample
        confidence: Confidence threshold (ignored in stub)
        minimum_reads_per_taxon: Minimum reads threshold per taxon
        metadata: Metadata DataFrame for filtering
        controls: List of control sample IDs
        report_id: Report ID (ignored in stub)
        
    Returns:
        Dictionary containing:
        - decontaminated_reads: Polars DataFrame with OTU data
        - valid_samples: List of valid sample IDs
    """
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    test_data_dir = script_dir.parent / "test_data"
    decontaminated_reads_path = test_data_dir / "decontaminated_reads.csv"

    decontaminated_reads = pl.read_csv(decontaminated_reads_path)
    valid_samples = ['TIDE_36', 'TIDE_37', 'TIDE_33', 'PP204C1', 'TIDE_35', 'PP204C2', 'TIDE_38', 'TIDE_34', 'TIDE_44', 'TIDE_45', 'TIDE_48', 'TIDE_42', 'TIDE_43', 'TIDE_39', 'TIDE_41', 'TIDE_40', 'TIDE_46', 'PPEXT', 'PPPCR', 'TIDE_EB', 'PP204B2', 'PP204B1', 'PP202B1', 'PP202A2', 'PP202B2', 'PP202C1', 'PP202C2', 'PP203A1', 'PP182C1', 'PP182A2', 'PP182B1', 'PP182B2', 'PP183A1', 'PP182C2', 'PP183A2', 'EMB_27', 'MPA_13', 'EMB_30', 'MPA_1', 'EMB_28', 'MPA_10', 'EMB_29', 'MPA_11', 'EMB_31', 'FB_EMB123', 'MPA_12', 'EMB_32', 'FB_MPA123', 'FB_TIDE123', 'PP11A1', 'MPA_2', 'MPA_3', 'MPA_9', 'MPA_7', 'MPA_8', 'MPA_16', 'MPA_6', 'MPA_4', 'MPA_5', 'MPA_14', 'PCR_blank', 'EMB_24', 'EMB_22', 'EMB_23', 'EMB_17', 'EMB_21', 'EMB_20', 'EMB_18', 'EMB_19', 'EMB_25', 'EMB_26', 'EB_EMB', 'PP16A1', 'PP16A2', 'PP16B1', 'PP16B2', 'PP16C1', 'PP14A2', 'PP14B1', 'PP15B1', 'PP14B2', 'PP14C2', 'PP15B2', 'PP15A2', 'PP15C2', 'PP14C1', 'PP15C1', 'PP15A1', 'TiDE_47', 'PP183B1', 'PP183B2', 'PP183C2', 'PP183C1', 'PP184A1', 'PP184A2', 'PP184C1', 'PP184B1', 'PP184B2', 'PP11A2', 'PP14A1', 'PP11B2', 'PP11B1', 'PP203A2', 'PP203B1', 'PP204A2', 'PP204A1', 'PP203C2', 'PP203C1', 'PP203B2', 'PP186C2', 'PP187A1', 'PP187C1', 'PP187B1', 'PP187A2', 'PP187B2', 'PP18A2', 'PP187C2', 'PP18B1', 'PP18A1', 'PP181A1', 'PP181A2', 'PP16C2', 'PP181B1', 'PP181B2', 'PP181C1', 'PP181C2', 'PP182A1', 'PP18B2', 'PP18C1', 'PP18C2', 'PP201B1', 'PP201A1', 'PP202A1', 'PP201B2', 'PP201C1', 'PP201A2', 'PP201C2', 'PP186A2', 'PP186A1', 'PP184C2', 'PP186B2', 'PP186B1', 'PP186C1']
    
    return {
        "decontaminated_reads": decontaminated_reads,
        "valid_samples": valid_samples,
    }


# Additional stub functions that might be needed

def get_species_list(session: object, species_list_id: str) -> pl.DataFrame:
    """Stub for species list retrieval."""
    # Return a simple species list for testing
    return pl.DataFrame({
        "name": [
            "Oncorhynchus mykiss",
            "Rhinichthys osculus", 
            "Baetis tricaudatus",
            "Chironomus plumosus",
            "Acroneuria abnormis",
            "Optioservus fastiditus"
        ]
    })


def get_labels(bucket: str, file_path: str) -> pl.DataFrame:
    """Stub for labels retrieval."""
    if "LabelsAndLegends" in file_path:
        return pl.DataFrame({
            "Environmental_Variable": ["bio01", "bio12", "elevation", "site", "temporal_months"],
            "Label": ["Temperature", "Precipitation", "Elevation", "Site", "Month"],
            "Axis_Label": ["Temperature (°C)", "Precipitation (mm)", "Elevation (m)", "Site", "Month"]
        })
    elif "Categories" in file_path:
        return pl.DataFrame({
            "Environmental_Variable": ["iucn_cat", "biome_type"],
            "value": ["II", "forest"],
            "description": ["National Park", "Forest Ecosystem"]
        })
    else:
        return pl.DataFrame()


def get_taxonomic_ranks(taxonomic_rank: str = "species") -> List:
    """Stub for taxonomic ranks."""
    all_ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    
    if taxonomic_rank == "taxonomic_path":
        return [all_ranks, None, len(all_ranks) - 1]
    
    try:
        rank_index = all_ranks.index(taxonomic_rank)
        return [all_ranks, None, rank_index]
    except ValueError:
        # Default to species if rank not found
        return [all_ranks, None, 6]