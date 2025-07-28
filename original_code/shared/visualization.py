"""
Visualization helper functions for beta diversity analysis.

This module contains utilities for creating visualizations, including
ellipse calculations and display formatting functions.
"""

import numpy as np


def int_to_letter(n: int) -> str:
    """
    Convert an integer to a letter (1 -> A, 2 -> B, etc.).

    Args:
        n: Integer to convert (1-based)

    Returns:
        Letter corresponding to the number
    """
    return chr(n + 64)


def point_on_ellipse(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle: float,
    t: float,
) -> tuple[float, float]:
    """
    Calculate a point on the rotated ellipse.

    Args:
        center_x: X coordinate of ellipse center
        center_y: Y coordinate of ellipse center
        width: Width of the ellipse
        height: Height of the ellipse
        angle: Rotation angle of the ellipse
        t: Parameter that determines position on ellipse (0 to 2π)

    Returns:
        Tuple of (x, y) coordinates on the ellipse
    """
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    x = center_x + (width / 2 * cos_t * cos_angle - height / 2 * sin_t * sin_angle)
    y = center_y + (width / 2 * cos_t * sin_angle + height / 2 * sin_t * cos_angle)

    return x, y
