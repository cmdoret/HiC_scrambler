# Functions which will be used for the generation of features linked to BAM files.

import numpy as np
from numpy.lib.stride_tricks import DummyArray
import pysam as ps

import pathlib
from typing import Tuple, Iterator

def check_gen_sort_index(bam: "pysam.AlignmentFile", cores: int = 1) -> str:
    """
    Index the input BAM file if needed. If the file is not coordinate-sorted,
    generate a sorted file. Returns the path to the sorted-indexed BAM file.
    Does nothing and return input file path if it was already indexed and
    coordinate-sorted.
    Parameters
    ----------
    bam : pysam.AlignmentFile
        Handle to the BAM file to index.
    cores : int
        Number of CPU cores to use for sorting.
    Returns
    -------
    sorted_bam : str
        The path to the sorted indexed BAM file
    """

    def check_bam_sorted(path: str) -> bool:
        """Checks whether target BAM file is coordinate-sorted"""
        header = ps.view(str(path), "-H")
        if header.count("SO:coordinate") == 1:
            issorted = True
        else:
            issorted = False
        return issorted

    bam_path = pathlib.Path(bam.filename.decode())

    try:
        # If the file has an index, there is nothing to do
        bam.check_index()
        sorted_bam = bam_path
    except ValueError:
        # Make a new sorted BAM file and store name for indexing
        if not check_bam_sorted(bam_path):
            sorted_bam = str(bam_path)
            print("Saving a coordinate-sorted BAM file as ", sorted_bam)
            ps.sort(
                str(bam_path), "-O", "BAM", "-@", str(cores), "-o", sorted_bam
            )
        else:
            sorted_bam = str(bam_path)
        # Index the sorted BAM file (input file if it was sorted)
        print("Indexing BAM file")
        ps.index("-@", str(cores), sorted_bam)

    return str(sorted_bam)

def parse_ucsc_region(ucsc: str) -> Tuple[str, int, int]:
    """Parse a UCSC-formatted region string into a triplet (chrom, start, end).

    Parameters
    ----------
    ucsc : str
        String representing a genomic region in the format chromosome:start-end.

    Returns
    -------
    tuple of (str, int, int) :
        The individual values extracted from the UCSC string.

    Examples
    --------
    >>> parse_ucsc_region('chrX:10312-31231')
    ('chrX', 10312, 31231)
    """
    try:
        chrom, bp_range = ucsc.split(":")
        start, end = bp_range.split("-")
        start, end = int(start), int(end)
    except ValueError:
        raise ValueError("Invalid UCSC string.")
    return chrom, start, end


def bam_region_read_ends(file: str, region: str, side: str = "both") -> np.ndarray:
    """Retrieves the number of read ends at each position for a BAM region.

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
    
    file_sorted = check_gen_sort_index(bam)
    
    bam = ps.AlignmentFile(file_sorted, "rb")

    chrom, start, end = parse_ucsc_region(region)


    start_arr = np.zeros(end - start)
    end_arr = np.zeros(end - start)
    
    for read in bam.fetch(chrom, start, end):
        
        
        if (read.reference_start - start) >=0:
            start_arr[read.reference_start - start] += 1

        if (read.reference_end - start) < end - start:           
            end_arr[read.reference_end - start] += 1


    if side == "start":
        return start_arr
    elif side == "end":
        return end_arr
    else:
        return start_arr, end_arr


def bam_region_coverage(file: str, region: str) -> np.ndarray:
    """Retrieves the basepair-level coverage track for a BAM region.

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

def mean_weighted(array : np.array) -> float:
    """Return a mean where coords from the center are more important than 
    at the start or at the beginning of the array.

    Parameters
    ----------
    array : np.array
        Array of the elements we want to mean.

    Returns
    -------
    float :
        Weighted mean of the array.
    """
    kernel_size = (len(array)//2)/(2*np.sqrt(np.log(10))) # To have at the beginning and at the end 0.01.
    if len(array)%2 ==0:

        tab_forward = np.arange(1,len(array)//2+1)
        tab_reverse = np.arange(len(array)//2, 0, -1)

        tab = np.concatenate((tab_reverse, tab_forward))
    
    if len(array)%2 ==1:

        tab_forward = np.arange(0,len(array)//2+1)
        tab_reverse = np.arange(len(array)//2, 0, -1)
        

        tab = np.concatenate((tab_reverse, tab_forward))
    tab = np.exp(-(tab**2)/(2*(kernel_size**2)))
    return np.mean(tab*array)