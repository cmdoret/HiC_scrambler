# Utilities to generate an artificial Hi-C matrix
import numpy as np


def powerlaw(dist, C, alpha):
    if distance == 0:
        return C
    return C * dist ** (-alpha)


def normal_bias(distance):
    return (1 + np.exp(-((distance - 20) ** 2) / (2 * 30))) * 4


# function to add bias to a matrix
def add_bias(M, slope, medianIF, powerlaw_alpha, biasFun=normal_bias):
    col_num = 0
    for i in range(M.shape[0]):
        for j in range(col_num, M.shape[1]):
            distance = j - i
            Bmean = powerlaw(distance, medianIF, powerlaw_alpha)
            new_value = np.round(M[i, j] * biasFunc(distance))
            new_value = 0 if new_value < 0 else new_value
            mat[i, j] = new_value
            mat[j, i] = mat[i, j]
        col_num = col_num + 1
    return mat


# WIP: Translate R code to python and modify
# simulate synthetic Hi-C contact maps.
# IF : interaction frequency
def sim_mat(
    nrow, ncol, medianIF, sdIF, powerlaw_alpha, sd_alpha, prop_zero_slope
):
    pass

