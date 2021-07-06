# Functions which are important in order to create features

import numpy as np
import pandas as pd

np.seterr(divide="ignore")


def count_0(matrix):
    """
    Counts the number of zeros for each row of the matrix.
    """
    counts = np.zeros(matrix.shape[0])
    for k in range(0, matrix.shape[0]):

        row = matrix[k, :]
        counts[k] = len(np.where(row == 0)[0])
    return counts


def white_index(matrix):
    """
    Returns the index of the rows of the matrix where there is a lot of zeros (white bands). 
    """
    size_mat = matrix.shape[0]
    percent = 0.9

    counts = count_0(matrix)

    return np.where(counts >= size_mat * percent)[0]


def sv_dataframe_modification(sv_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Modify dataframe because some informations are not complete or exactly correct (INS, DEL...).
    """

    # BP3 useless for sv which are not TRA
    indices_not_tra = sv_dataframe["sv_type"] != "TRA"

    bp3 = sv_dataframe["breakpoint3"].values
    coord_bp3 = sv_dataframe["coord_bp3"].values

    bp3[indices_not_tra] = -1
    coord_bp3[indices_not_tra] = -1

    sv_dataframe["breakpoint3"] = bp3
    sv_dataframe["coord_bp3"] = coord_bp3

    ### Modication name of sv_type
    sv_types = sv_dataframe["sv_type"].values

    # Our DEL are INS
    indices_DEL = sv_dataframe["sv_type"] == "DEL"

    sv_types[indices_DEL] = "INS"


    sv_dataframe["sv_type"] = sv_types

    return sv_dataframe
