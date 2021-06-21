# Functions which are important in order to create features

import numpy as np
import BAM_functions as bm
from RepeatsFinder import RepeatsFinder
np.seterr(divide = 'ignore') 

def count_0(matrix):
    """
    Counts the number of zeros for each row of the matrix.
    """
    counts = np.zeros(matrix.shape[0])
    for k in range(0, matrix.shape[0]):

        row = matrix[k,:]
        counts[k] = len(np.where(row == 0)[0])
    return counts


def white_index(matrix):
    """
    Returns the index of the rows of the matrix where there is a lot of zeros (white bands). 
    """
    size_mat = matrix.shape[0]
    percent = 0.9

    counts = count_0(matrix)
    
    return np.where(counts >= size_mat*percent)[0]


def create_features(path : str, chrom : str):
    """
    Create features in order to make the detection of the exact position of a SV. Firstly it creates features
    of coords where there is SVs. After that it will creates a lot of features of coords where there is no SVs.

    Parameters
    ----------
    path : str
        Path where all the informations created before are. It will use them to create the features.
    chrom : str
        Chromosome where modifications have been done.
        
    Returns
    -------
    bool :
        Returns a boolean. True if it is a repeat, False it is not. 
    """
    size_train_forest = 15
    size_win_bp = 2
    binsize = 2000

    RFinder = RepeatsFinder()

    starts = np.load(path + "/start_reads.npy")
    ends = np.load(path + "/end_reads.npy")
    labels = np.load(path + "/y.npy")

    starts = starts[labels != 0]
    ends = ends[labels != 0]

    coordsBP = np.load(path + "/coordsBP.npy")


    scrambled = np.load(path + "/scrambled.npy")
    scrambled = np.log10(scrambled)
    scrambled[scrambled == -np.inf] = 0

    white_inds = white_index(scrambled)

    ind_beg = 64
    ind_end = len(scrambled) - 64
    
    valid_coords = np.where((coordsBP >= ind_beg*binsize) & (coordsBP <= ind_end*binsize))
    coords_used = coordsBP[valid_coords]

    mean_start_reads = list()
    mean_end_reads = list()
    coverages = list()
    repeats = list()
    labels = list()

    for coord in coords_used:
        c_beg = int(coord - size_train_forest//2)
        c_end = int(coord + size_train_forest//2)

        
        region = chrom + ":" + str(c_beg-size_win_bp//2) + "-" + str(c_end+ size_win_bp//2 + 1)

        start_reads, end_reads = bm.bam_region_read_ends(file = path + "/scrambled.for.bam", region = region, side  = "both")
        coverage = bm.bam_region_coverage(file = path + "/scrambled.for.bam", region = region)

        mean_start_reads_inter = (start_reads + np.concatenate((start_reads[1:], np.zeros(1))) + np.concatenate((np.zeros(1), start_reads[:len(start_reads)-1])))[1:-1]//3

        mean_end_reads_inter = (end_reads + np.concatenate((end_reads[1:], np.zeros(1))) + np.concatenate((np.zeros(1), end_reads[:len(end_reads)-1])))[1:-1]//3
                



        mean_start_reads.append((mean_start_reads_inter - np.concatenate((mean_start_reads_inter[1:], np.zeros(1))))[:len(mean_start_reads_inter)-1])
        mean_end_reads.append((mean_end_reads_inter- np.concatenate((mean_end_reads_inter[1:], np.zeros(1))))[:len(mean_end_reads_inter)-1])
        coverages.append(coverage)

        repeats.append(RFinder.predict(coord, path, chrom, verbose = False))
        labels.append(1)

    
    coords_used = coords_used//binsize
    coords_used = np.concatenate((coords_used, white_inds))

    test = False
    while test == False:

        index = np.random.randint(ind_beg+2, ind_end-2)
        test = len(np.where(coords_used//binsize == index)[0]) == 0

    coords_used = list(coords_used)
    coords_used.append(index)
    coords_used = np.array(coords_used)

    

    for coord in range(index*binsize, (index+1)*binsize, 10):
        
        c_beg = int(coord - size_train_forest//2)
        c_end = int(coord + size_train_forest//2)


        region = chrom + ":" + str(c_beg-size_win_bp//2) + "-" + str(c_end+ size_win_bp//2 + 1)
        
        start_reads, end_reads = bm.bam_region_read_ends(file = path + "/scrambled.for.bam", region = region, side  = "both")
        coverage = bm.bam_region_coverage(file = path + "/scrambled.for.bam", region = region)

        mean_start_reads_inter = (start_reads + np.concatenate((start_reads[1:], np.zeros(1))) + np.concatenate((np.zeros(1), start_reads[:len(start_reads)-1])))[1:-1]//3

        mean_end_reads_inter = (end_reads + np.concatenate((end_reads[1:], np.zeros(1))) + np.concatenate((np.zeros(1), end_reads[:len(end_reads)-1])))[1:-1]//3


        mean_start_reads.append((mean_start_reads_inter - np.concatenate((mean_start_reads_inter[1:], np.zeros(1))))[:len(mean_start_reads_inter)-1])
        mean_end_reads.append((mean_end_reads_inter- np.concatenate((mean_end_reads_inter[1:], np.zeros(1))))[:len(mean_end_reads_inter)-1])

        coverages.append(coverage)

        labels.append(0)
        repeats.append(RFinder.predict(coord, path, chrom, verbose = False))
           
    repeats = np.array(repeats).reshape((-1,1))
    mean_start_reads = np.array(mean_start_reads)
    mean_end_reads = np.array(mean_end_reads)
    coverages = np.array(coverages)

    return mean_start_reads, mean_end_reads, labels, repeats, coverages
