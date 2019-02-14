import numpy as np


def powerlaw(dist, C, alpha):
    if distance == 0:
        return C
    return C * dist**(-alpha)


def normal_bias(distance):
    return (1 + np.exp(- ((distance - 20) ** 2)/(2 * 30))) * 4


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
    return(mat)


# simulate synthetic Hi-C contact maps.
# IF : interaction frequency
def sim_mat(nrow=100, ncol=100, medianIF, sdIF, powerlaw_alpha,
            sd_alpha, prop_zero_slope):
    pass
# add CNV
def add_CNV(mat, CNV_location, CNV_proportion, CNV.multiplier):
  loc = CNV_location[0]:CNV_location[1]
  idx = expand.grid(loc, loc)
  n.idx <- nrow(idx)
  new.idx <- sample(1:n.idx, size = round(CNV.proportion * n.idx), replace = FALSE)
  new.idx <- idx[new.idx,]
  new.idx <- as.matrix(new.idx)
  mat[new.idx] <- mat[new.idx] * CNV.multiplier
  # force symmetry
  mat[lower.tri(mat)] <- t(mat)[lower.tri(mat)]
  return(mat)
}

