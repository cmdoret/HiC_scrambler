from typing import Iterable
import numpy as np

# Utilities to apply SV to a genome


def deletion(start: int, end: int, genome: str) -> str:
    """Apply deletion on input genome.

    Examples
    --------
    >>> deletion(3, 8, "ACGTACGTACGT")
    'ACGACGT'
    """
    mutseq = genome[:start] + genome[end:]
    return mutseq


def inversion(genome: str) -> str:
    """Apply deletion on input genome."""
    return ...


def update_coords_del(start: int, end: int, coords: Iterable[int]) -> "np.ndarray[int]":
    """
    Update coordinates after applying a deletion at specified positions

    Examples
    --------
    >>> update_coords_del(12, 18, [4, 15, 22])
    array([ 4, 12, 16])
    """
    # Shift coordinates on the right of DEL region
    del_size = end - start
    coords = np.array(coords)
    coords_edit = coords[coords > start]
    coords[coords > start] = coords_edit - np.minimum(del_size, coords_edit - start)
    coords[coords < 0] = 0

    return coords


def update_coords_inversion(
    del_start: int, del_end: int, coords: Iterable[int]
) -> "np.ndarray[int]":
    """Update coordinates after applying a deletion at specified positions"""
