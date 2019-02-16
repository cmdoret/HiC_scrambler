# Trying a simple ML model to predict Hi-C patterns
# cmdoret, 20190131

import numba as nb
import numpy as np
from Bio import SeqIO


@nb.njit(error_model='numpy', parallel=True)
def subset_mat(matrix, coords, labels, winsize=128):
    """
    Samples evenly sized windows from a matrix. Windows are centered around
    input coordinates. Windows and their associated labels are returned.
    Parameters
    ----------
    matrix : numpy.ndarray of floats
        The Hi-C matrix as a 2D array.
    coords : numpy.ndarray of ints
        Pairs of coordinates for which subsets should be generated. A window
        centered around each of these coordinates will be sampled. Dimensions
        are [N, 2].
    labels : numpy.ndarray of ints
        1D array of labels corresponding to the coords given.
    winsize : int
        Size of windows to sample from the matrix.
    Returns
    -------
    x : numpy.ndarray of floats
        The 3D feature vector to use as input in a keras model.
        Dimensions are [N, winsize, winsize].
    y : numpy.array of ints
        The 1D label vector of N values to use as prediction in a keras model.
    """

    h, w = matrix.shape
    i_w = (h - winsize / 2)
    j_w = (w - winsize / 2)

    # Only keep coords far enough from borders
    valid_coords = np.where((coords[:, 0] > winsize / 2) &
                            (coords[:, 1] > winsize / 2) &
                            (coords[:, 0] < i_w) &
                            (coords[:, 1] < j_w))[0]
    coords = coords[valid_coords, :]
    # Number of windows to generate
    n_windows = coords.shape[0]
    x = np.zeros((n_windows, winsize, winsize), dtype=np.float64)
    y = np.zeros(n_windows, dtype=np.int64)
    if winsize <= min(h, w):
        print("Window size must be smaller than the Hi-C matrix.")

    halfw = winsize / 2
    for i in nb.prange(n_windows):
        c = coords[i, :]
        x[i, :] = matrix[(c[0] - halfw): (c[0] + halfw),
                         (c[1] - halfw): (c[1] + halfw)]
        y[i] = labels[i]

    return x, y


def edit_genome(fasta_in, fasta_out, sv_dict):
    """
    Given a fasta file and a dictionary of structural variants and their
    positions, generate a new genome by applying the input changes.
    Parameters
    ----------
    fasta_in : str
        Path to the input genome in fasta format.
    fasta_out : str
        Path where the edited genome will be written in fasta format.
    sv_dict : dict
        Dictionary with names of structural variants as keys and tuples
        of chromosome names and coordinates as values. Currently, the only
        structural variants supported are inversions.
    Returns
    -------
    """
    with open(fasta_out, 'w') as fa_out:
        for chrom in SeqIO.parse(fasta_in, format='fasta'):
            for sv_type, coords in sv_dict:
                if sv_type == "INV":
                    for sv in coords:
                        start, end = sv[1]
                        if sv[0] == chrom.id:
                            chrom.seq[start: end] = chrom.seq[end:start]
        SeqIO.write(chrom, fa_out)


def generate_sv(
        chromsizes,
        sv_freqs={"INV": 0.01},
        params={"INV": 2250, "DEL": 3100},
                ):
    """
    Generates random structural variations, given a dictionary of SV types
    and their frequencies, as well as a dictionary of chromosome names and
    their respective lengths.
    Parameters
    ----------
    chromsizes : dict
        Dictionary with chromosome names as keys and their length in basepairs
        as values.
    sv_freqs : dict
        Dictionary with event types as keys (such as INV, DEL, INS, TRA)
        and their frequencies, in #event / kb.
    params : dict
        Dictionary with SV types as keys and their characteristics as values.
    For inversions and deletions, mean length is used. (chosen
    approximately based on Sudmant et al, Science, 2015).
    Returns
    -------
    list :
        A list where each element is an SV. Each SV is a tuple of
        (chrom, (coords)).
    """
    # Relative abundance of each event type (placeholder values)
    rel_abun = {"INV": 8, "DEL": 400, "DUP": 60, "INS": 160, "CNV": 350}
    for chrom, size in chromsizes:
        n_sv = size * sum(sv_freqs.values()) / 1000
        out_sv = [None * n_sv]
        sv_count = 0
        for sv_type in sv_freqs:
            prop_event = rel_abun[sv_type] / \
                sum([n for s, n in rel_abun if s in sv_freqs])
            n_event = n_sv * prop_event
            for _ in n_event:
                start = np.random.randint(size)
                end = start + np.random.normal(loc=params[sv_type],
                                               scale=0.1 * params[sv_type])
                out_sv[sv_count] = (chrom, (start, end))
                sv_count += 1

    return out_sv
