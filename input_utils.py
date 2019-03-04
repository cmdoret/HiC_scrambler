# Trying a simple ML model to predict Hi-C patterns
# cmdoret, 20190131

import numpy as np
import pandas as pd
from Bio import SeqIO, Seq
from hicstuff import view as hcv


def load_chromsizes(path):
    """Loads hicstuff info_contig file into a dictionary"""
    chr_names = np.loadtxt(path, usecols=(0,), skiprows=1, dtype=str, ndmin=1)
    chr_len = np.loadtxt(path, usecols=(1,), skiprows=1, dtype=np.int64, ndmin=1)
    chromsizes = {n: l for n, l in zip(chr_names, chr_len)}
    return chromsizes


def subset_mat(matrix, coords, labels, winsize=128, prop_negative=0.5):
    """
    Samples evenly sized windows from a matrix. Windows are centered around
    input coordinates. Windows and their associated labels are returned.
    Parameters
    ----------
    matrix : scipy.sparse.coo_matrix
        The Hi-C matrix as a 2D array in sparse format.
    coords : numpy.ndarray of ints
        Pairs of coordinates for which subsets should be generated. A window
        centered around each of these coordinates will be sampled. Dimensions
        are [N, 2].
    labels : numpy.ndarray of ints
        1D array of labels corresponding to the coords given.
    winsize : int
        Size of windows to sample from the matrix.
    prop_negative : float
        The proportion of windows without SVs desired. If set to 0.5, when given
        a list of 23 SV, the function will output 46 observations (windows); 23
        without SV (picked randomly in the matrix) and 23 with SV.
    Returns
    -------
    x : numpy.ndarray of floats
        The 3D feature vector to use as input in a keras model.
        Dimensions are [N, winsize, winsize].
    y : numpy.array of ints
        The 1D label vector of N values to use as prediction in a keras model.
    """

    h, w = matrix.shape
    i_w = h - winsize // 2
    j_w = w - winsize // 2
    sv_to_int = {"INV": 1, "DEL": 2, "INS": 3}
    # Only keep coords far enough from borders of the matrix
    valid_coords = np.where(
        (coords[:, 0] > winsize / 2)
        & (coords[:, 1] > winsize / 2)
        & (coords[:, 0] < i_w)
        & (coords[:, 1] < j_w)
    )[0]
    coords = coords[valid_coords, :]
    labels = labels[valid_coords]
    # Number of windows to generate (including negative windows)
    n_windows = int(coords.shape[0] // (1 - prop_negative))
    x = np.zeros((n_windows, winsize, winsize), dtype=np.float64)
    y = np.zeros(n_windows, dtype=np.int64)
    if winsize >= min(h, w):
        print("Window size must be smaller than the Hi-C matrix.")
    matrix = matrix.tocsr()
    halfw = winsize // 2
    # Getting SV windows
    for i in range(coords.shape[0]):
        c = coords[i, :]
        win = matrix[(c[0] - halfw) : (c[0] + halfw), (c[1] - halfw) : (c[1] + halfw)]
        x[i, :, :] = hcv.sparse_to_dense(win, remove_diag=False)
        y[i] = sv_to_int[labels[i]]
    # Getting negative windows
    neg_coords = set()
    for i in range(coords.shape[0], n_windows):
        tries = 0
        c = np.random.randint(winsize // 2, i_w)
        # this coordinate must not exist already
        while (c in coords[:, 0]) or (c in neg_coords) :
            print("{} is already used. Trying another position...".format(c))
            # If unable to find new coords, just return output until here
            if tries > 100:
                return x[:i, :, :], y[:i]
            c = np.random.randint(winsize // 2, i_w)
            neg_coords.add(c)
            tries += 1
        win = matrix[(c - halfw) : (c + halfw), (c - halfw) : (c + halfw)]
        x[i, :, :] = hcv.sparse_to_dense(win, remove_diag=False)
        y[i] = 0
    return x, y


def edit_genome(fasta_in, fasta_out, sv_df):
    """
    Given a fasta file and a dataframe of structural variants and their
    positions, generate a new genome by applying the input changes.
    Parameters
    ----------
    fasta_in : str
        Path to the input genome in fasta format.
    fasta_out : str
        Path where the edited genome will be written in fasta format.
    sv_df : dict
        Dictionary with names of structural variants as keys and tuples
        of chromosome names and coordinates as values. Currently, the only
        structural variants supported are inversions.
    """
    with open(fasta_out, "w") as fa_out:
        for chrom in SeqIO.parse(fasta_in, format="fasta"):
            mutseq = Seq.MutableSeq(str(chrom.seq))
            for row_num in range(sv_df.shape[0]):
                row = sv_df.iloc[row_num, :]
                sv_type = row.sv_type
                chr, start, end = row.chrom, int(row.start), int(row.end)
                if sv_type == "INV":
                    if chr == chrom.id:
                        mutseq[start:end] = mutseq[end - 1 : start - 1 : -1]
            chrom = SeqIO.SeqRecord(seq=mutseq, id=chrom.id, description="")
            SeqIO.write(chrom, fa_out, format="fasta")


def generate_sv(
    chromsizes, sv_types={"INV"}, freq=10e-4, params={"INV": 2250, "DEL": 3100}
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
    pandas.DataFrame :
        A dataframe where each row is a SV. columns represent
        sv_type, chrom, start, end.
    """
    # Relative abundance of each event type (placeholder values)
    rel_abun = {"INV": 8, "DEL": 400, "DUP": 60, "INS": 160, "CNV": 350}
    print(chromsizes)
    for chrom, size in chromsizes.items():
        n_sv = size * freq
        out_sv = pd.DataFrame(np.empty((int(n_sv), 4)))
        out_sv.columns = ["sv_type", "chrom", "start", "end"]
        sv_count = 0
        for sv_type in sv_types:
            prop_event = rel_abun[sv_type] / sum(
                [n for s, n in rel_abun.items() if s in sv_types]
            )
            # Number of a given event dictated by relative freqency of each SV
            # type and total SV freq desired
            n_event = int(n_sv * prop_event)
            for _ in range(n_event):
                # Start position is random and length is picked from a normal
                # distribution centered around mean length.
                start = np.random.randint(size)
                end = start + np.random.normal(
                    loc=params[sv_type], scale=0.1 * params[sv_type]
                )
                out_sv.iloc[sv_count, :] = (sv_type, chrom, start, end)
                sv_count += 1

    return out_sv


def pos_to_coord(sv_df, frags_df, bin_size):
    """
    Converts start - end genomic positions from structural variations to breakpoints
    in matrix coordinates.
    Parameters
    ----------
    sv_df : pandas.DataFrame
        A dataframe containg the type and genomic start-end coordinates of
        strucural variations as given by generate_sv().
    frags_df : pandas.DataFrame
        A dataframe containing the list of fragments of bins in the Hi-C matrix.
    bin_size : int
        The bin size used for the matrix.
    Returns
    -------
    breakpoints : numpy.array of int
        A N x 2 numpy array of numeric values representing X, Y coordinates of structural
        variations breakpoints in the matrix.
    labels : numpy.array of str
        An N X 1 array of labels corresponding to SV type.
    """
    # Get coordinates to match binning
    sv_df.start = (sv_df.start // bin_size) * bin_size
    sv_df.end = (sv_df.end // bin_size) * bin_size
    # Put start and end in the same column, 1 row / breakpoint
    s_df = sv_df.loc[:, ["sv_type", "chrom", "start"]]
    s_df.rename(index=str, columns={"start": "pos"}, inplace=True)
    e_df = sv_df.loc[:, ["sv_type", "chrom", "end"]]
    e_df.rename(index=str, columns={"end": "pos"}, inplace=True)
    sv_df = pd.concat([s_df, e_df]).reset_index(drop=True)
    # Assign matrix coordinate (fragment index) to each breakpoint
    frags_df["coord"] = frags_df.index
    sv_frags = sv_df.merge(
        frags_df, left_on=["chrom", "pos"], right_on=["chrom", "start_pos"], how="left"
    )
    breakpoints = np.vstack([sv_frags.coord, sv_frags.coord]).T
    breakpoints.astype(int)
    labels = np.array(sv_frags.sv_type.tolist())
    return breakpoints, labels
