"""Geometry utilities for RNA backbone analysis.

Pure-numpy helpers for distances, angles, and torsion (dihedral) angles.
These operate on (3,) numpy arrays, not custom XYZ classes.
"""

from __future__ import annotations

import math

import numpy as np

PI = math.pi
MIN_ZERO_TOLERANCE = 1e-7


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 3-D points."""
    return float(np.linalg.norm(a - b))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle A-B-C in radians (vertex at B)."""
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + MIN_ZERO_TOLERANCE)
    return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))


def torsion_angle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    """Dihedral (torsion) angle A-B-C-D in radians, range (-π, π]."""
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    norm1 = np.linalg.norm(n1)
    norm2 = np.linalg.norm(n2)
    if norm1 < MIN_ZERO_TOLERANCE or norm2 < MIN_ZERO_TOLERANCE:
        return 0.0
    cos_val = np.dot(n1, n2) / (norm1 * norm2)
    ang = float(np.arccos(np.clip(cos_val, -1.0, 1.0)))
    sign = 1.0 if np.dot(b2, np.cross(n1, n2)) >= 0 else -1.0
    return sign * ang


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / PI


def deg_to_rad(deg: float) -> float:
    return deg * PI / 180.0


def rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """RMSD between two (N, 3) coordinate arrays."""
    diff = coords_a - coords_b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
