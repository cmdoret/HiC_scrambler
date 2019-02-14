# Trying a simple ML model to predict Hi-C patterns
# cmdoret, 20190131

import numba as nb
import numpy as np


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
    out = np.empty((i_w, j_w))
    for i in nb.prange(n_windows):
        c = coords[i, :]
        x[i, :] = matrix[(c[0] - halfw): (c[0] + halfw),
                         (c[1] - halfw): (c[1] + halfw)]
        y[i] = labels[i]

    return out
