"""
Text formatting and display helper functions for beta diversity analysis.

This module contains utilities for formatting parameter names, labels,
and other text-based display elements.
"""


def humanize_parameter_name(param_name: str) -> str:
    """
    Convert a parameter name like 'temporal_months' to a human-readable format like 'Temporal Months'.

    Args:
        param_name: The parameter name to humanize (e.g., 'temporal_months')

    Returns:
        Human-readable version of the parameter name (e.g., 'Temporal Months')
    """
    if not param_name:
        return param_name

    # Replace underscores with spaces and capitalize each word
    return param_name.replace("_", " ").title()
