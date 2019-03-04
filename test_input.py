#!/bin/env python3
# Unit tests related to input generation
# cmdoret, 20190302
import input_utils as iu
import pytest
import numpy as np


MAT_SLICES_PARAM = ("matrix_size,window_size", [5, 10, 20, 50, 100], [2, 3, 10, 1, 20])


@pytest.mark.parametrize(*MAT_SLICES_PARAM)
def test_subset_mat(matrix_size, window_size):
    mat = np.random.random((matrix_size, matrix_size))
    mat += mat.T
    h, w = mat.shape
    i_w = h - window_size / 2
    j_w = w - window_size / 2
    # Pick an arbitrary number of random coordinates in the matrix
    coords = np.random.randint(0, matrix_size, (np.random.randint(matrix_size // 2), 2))
    labels = np.random.randint(0, 2, coords.shape[0])
    x, y = iu.subset_mat(mat, coords, labels, window_size)

    # Only keep coords far enough from borders of the matrix
    valid = np.where(
        (coords[:, 0] > window_size / 2)
        & (coords[:, 1] > window_size / 2)
        & (coords[:, 0] < i_w)
        & (coords[:, 1] < j_w)
    )[0]
    # valid_coords = coords[valid, :]
    valid_labels = labels[valid, :]

    assert x.shape[0] == y.shape[0] == valid_labels.shape[0]
    assert x.shape[1] == x.shape[2] == window_size
    assert (y == valid_labels).all()


def test_edit_genome():
    ...


def test_generate_sv():
    ...
