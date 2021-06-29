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


def translocation(start_cut: int, end_cut: int, start_paste: int, genome: str) -> str:
    """Apply translocation on input genome.

    Examples
    --------
    >>> translocation(2, 4, 7,"ACGTACGTACGT")
    'ACACGGTTACGT'
    """

    tra_size = end_cut - start_cut

    seq_cut = genome[start_cut:end_cut]

    mutseq = genome[:start_cut] + genome[end_cut:]

    if start_paste >= end_cut:
        start_paste -= tra_size  # Update coords

    mutseq = mutseq[:start_paste] + seq_cut + mutseq[start_paste:]

    return mutseq


def update_coords_del_before(
    start: int, end: int, coords: Iterable[int]
) -> "np.ndarray[int]":
    """
    Update coordinates after applying a deletion at specified positions.

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


def update_coords_del_after(
    start: int, end: int, coords: Iterable[int]
) -> "np.ndarray[int]":
    """
    For sv which have not been applied, the len of the sequence change so we must update the coords also.

    Examples
    --------
    >>> update_coords_del(12, 18, [4, 15, 22])
    array([ 4, 12, 16])
    """
    # Shift coordinates on the right of DEL region
    del_size = end - start
    coords = np.array(coords)
    coords -= del_size
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

    if (
        start + end
    ) % 2 == 1:  # During updating, if start+ end is impair, coords are shifted by one

        coords[(coords == start - 1) | (coords == end - 1)] += 1

    return coords


def update_coords_tra(
    start_cut: int, end_cut: int, start_paste: int, coords: Iterable[int]
) -> "np.ndarray[int]":
    """
    Update coordinates after applying a deletion at specified positions

    Examples
    --------
    >>> update_coords_del(12, 18, [4, 15, 22])
    array([ 4, 12, 16])
    """

    coords = np.array(coords)

    min_SV_breakpoint = min(start_cut, end_cut, start_paste)
    max_SV_breakpoint = max(start_cut, end_cut, start_paste)
    inter_SV_breakpoint = sorted([start_cut, end_cut, start_paste])[
        1
    ]  #  Intermediate breakpoint

    coords_before_inter = (coords < inter_SV_breakpoint) & (coords >= min_SV_breakpoint)
    coords_after_inter = (coords >= inter_SV_breakpoint) & (coords < max_SV_breakpoint)

    coords[coords_before_inter] += max_SV_breakpoint - inter_SV_breakpoint
    coords[coords_after_inter] -= inter_SV_breakpoint - min_SV_breakpoint

    return coords


def update_sgn_inversion(
    start: int,
    end: int,
    sgn_start: str,
    sgn_end: str,
    coords: Iterable[int],
    sgns: Iterable[str],
) -> "np.ndarray[str]":

    coords_inside_fragment = (coords > start) & (coords < end)
    sgn_inside_fragment = sgns[np.argsort(coords[coords_inside_fragment])]

    sgn_inside_fragment = sgns[coords_inside_fragment].values

    sgn_inside_fragment = np.array([sgn[::-1] for sgn in sgn_inside_fragment])

    sgns[coords_inside_fragment] = sgn_inside_fragment

    sgns[coords == start] = sgn_start[0] + sgn_end[0]
    sgns[coords == end] = sgn_start[1] + sgn_end[1]

    return sgns


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
