"""
Stub implementations for testing the challenge code.

This module provides stubbed versions of external dependencies
to allow developers to test their refactored code without needing
the full database and cloud infrastructure.
"""

from .metadata_stub import (
    process_metadata,
    load_reads_for_primer,
    get_species_list,
    get_labels,
    get_taxonomic_ranks,
)

__all__ = [
    "process_metadata",
    "load_reads_for_primer", 
    "get_species_list",
    "get_labels",
    "get_taxonomic_ranks",
]