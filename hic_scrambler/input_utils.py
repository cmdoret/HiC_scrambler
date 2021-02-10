"""
Utilities to preprocess input data before feeding it to the various models.
cmdoret, 202102010
"""

import numpy as np
import pysam as ps


def matrix_diag_chunks(matrix, size=128, stride=1):
    """
    Given an input matrix, yield a generator of chunks along the diagonal.
    Only compatible with symmetric matrices.
    """
    m, n = matrix.shape
    if m != n:
        raise ValueError("Input matrix must be square.")
    for i in range(0, matrix.shape - (size - 1), stride):
        start = i
        end = start + size
        chunk = matrix[start:end, start:end]
        yield chunk


def matrix_tiles(matrix, size=128, stride=1):
    """
    Chunk matrix into a grid of overlapping tiles and yield. Yield tiles and
    their coordinates. Works on asymmetric matrices.
    """
    for i in range(0, matrix.shape[0] - (size - 1), stride):
        start_x, end_x = i, i + size
        for j in range(0, matrix.shape[1] - (size - 1), stride):
            start_y, end_y = j, j + size
            chunk = matrix[start_x:end_x, start_y:end_y]
            yield i, j, chunk


def bam_region_coverage(file, region):
    """Retrieves the basepair-level coverage track for a BAM region

    Parameters
    ----------
    file : str
        path to the sorted, indexed BAM file.
    region : str
        UCSC-formatted region string (e.g. chr1:103-410)
    
    Returns
    -------
    numpy.ndarray of ints :
        number of reads overlapping each position in region.
    """

    bam = ps.AlignmentFile(file, "rb")
    chrom, start, end = parse_ucsc_region(region)

    cov_arr = np.zeros(end - start)
    for i, col in enumerate(bam.pileup(chrom, start, end, truncate=True)):
        cov_arr[i] = col.n

    return cov_arr


def bam_region_read_ends(file, region, side="both"):
    """ Retrieves the number of read ends at each position for a BAM region

    Parameters
    ----------
    file : str
        path to the sorted, indexed BAM file.
    region : str
        UCSC-formatted region string (e.g. chr1:103-410)
    side : str
        What end of the reads to count: left, right or both.
    
    Returns
    -------
    numpy.ndarray of ints :
        counts of read extremities at each position in region.
    """

    bam = ps.AlignmentFile(file, "rb")
    chrom, start, end = parse_ucsc_region(region)

    start_arr = np.zeros(end - start)
    end_arr = np.zeros(end - start)
    for read in enumerate(bam.fetch(chrom, start, end)):
        start_arr[read.reference_start - start] += 1
        end_arr[read.reference_end - start] += 1

    if side == "left":
        return start_arr
    elif side == "right":
        return end_arr
    else:
        return start_arr + end_arr


def parse_ucsc_region(ucsc):
    """Parse a UCSC-formatted region string into a triplet (chrom, start, end)"""
    try:
        chrom, bp_range = ucsc.split(":")
        start, end = bp_range.split("-")
        start, end = int(start), int(end)
    except ValueError:
        raise ValueError("Invalid UCSC string.")
    return chrom, start, end
