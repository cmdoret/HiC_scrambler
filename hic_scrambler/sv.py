from typing import Iterable
import numpy as np
import warnings
from Bio import Seq
warnings.filterwarnings("ignore") 
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
    """Apply inversion on input genome.

    Examples
    --------
    >>> inversion("ACGTACGTACGT")
    'TGCATGCATGCA'
    """    
    mutseq = Seq.reverse_complement(genome)
    return mutseq


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


def update_coords_inv(start: int, end: int, coords: Iterable[int]) -> "np.ndarray[int]":
    """Update coordinates after applying an inversion at specified positions
    
    Examples
    --------
    >>> update_coords_inv(12, 20, [4, 18, 22])
    array([ 4, 14, 16])
    """
    mid = (end + start) // 2
    coords = np.array(coords)
    coords_edit = coords[(coords >= start) & (coords <= end)]
    coords[(coords >= start) & (coords <= end)] = mid + mid - coords_edit
    return coords
	
def swap(starts: Iterable[int], ends: Iterable[int]) -> "np.ndarray[int]":
    """Swap start and end if end is lower than start
    
    Examples
    --------
    >>> update_coords_inv([4, 18, 22], [7, 13, 19])
    array([ 4, 13, 19])
    """
    swap_mask = starts > ends
    starts_old = np.copy(starts[swap_mask])
    ends_old = np.copy(ends[swap_mask])
    starts[swap_mask] = ends_old
    ends[swap_mask] = starts_old
    return starts, ends
