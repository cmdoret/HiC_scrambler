# Trying a simple ML model to predict Hi-C patterns
# cmdoret, 20190131

import matplotlib.pyplot as plt
import tensorflow as tf
import numba as nb


@nb.njit(error_model='numpy', parallel=True)
def subset_mat(matrix, coords, winsize=128):
    """
    Samples evenly sized windows from the matrix and gives them labels 0 or 1.
    If N coords are given, 2N windows will be sampled. Of which N will be
    centered on one of the coords and  given label 1, and N will be sampled
    randomly elsewhere in the matrix.
    Parameters
    ----------
    matrix : numpy.ndarray of floats
        The Hi-C matrix as a 2D array.
    coords : numpy.ndarray of ints
        Pairs of coordinates at which there are positive labels. A window
        centered around each of these coordinates will be sampled. Dimensions
        are [N, 2].
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

    h, w = img.shape
    i_w = (h - winsize)
    j_w = (w - winsize)
    x = np.zeros((N, winsize, winsize), dtype=np.float64)
    y = np.zeros(N, dtype=np.int64)
    if winsize <= min(h, w):
        print("Window size must be smaller than the Hi-C matrix.")

      out = np.empty((i_w,j_w))
      for i in nb.prange(0, i_w):
          for j in range(0, j_w):
              out[i,j]=np.std(img[i*step:i*step+N, j*step:j*step+N])
      return out

