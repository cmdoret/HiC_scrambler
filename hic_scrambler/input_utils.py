"""
Utilities to generate training data for the SV detecting NN.
cmdoret, 20190131
"""

import numpy as np
import pandas as pd
import cooler
from Bio import SeqIO, Seq
from hicstuff import view as hcv
import json


class GenomeMixer(object):
    """
    Handles genome edition through different types of structural variations.
    Note that the editions made in the genome are be the opposite of what their
    name says. The SV generated in the genome will be "mirrored" in the library.
    For example, deleting a region in the genome is akin to inserting a region
    in the library.
    
    Examples
    --------
        mix = GenomeMixer("genome.fasta", "config.json", "profile="Dmel")
        mix.generate_sv()
        mix.edit_genome("new_genome.fasta")
    
    Attributes
    ----------
    genome_path : str
        Path to the input genome to mix.
    config_path : str
        Path to the JSON config file to use for generating SVs.
    config : dict
        SV properties in a nested dictionary, loaded from a single profile in
        the config file.
    chromsizes : dict
        Contains the size of each chromosome in the format {chrom: len, ...}.
    sv : pandas.DataFrame
        Contains all structural variants, once generated by `generate_sv()`

    """

    def __init__(self, genome_path, config_path, config_profile=None):
        self.config_path = config_path
        self.config = self.load_profile(profile=config_profile)
        self.genome_path = genome_path
        self.chromsizes = self.load_chromsizes(self.genome_path)
        self.sv = None

    def load_profile(self, profile=None):
        """
        Load SV profile from a JSON config file. The top level JSON object is a
        profile. A config file can have multiple profiles, but only one will be
        loaded. Each profile contains the 'SV_freq' property, which gives the
        number of SV per bp, and the `SV_types` object. The SV_types object
        contains one object per SV type. Each SV contains multiple properties.

        Parameters
        ----------
        profile : str
            Name of the profile to load in the config file.

        Returns
        -------
        dict :
            A dictionary of SV types structured like 
            {"sv_type":{"property": value, ...}, ...}
        """
        config = json.load(open(self.config_path, "r"))
        if not profile:
            if len(config) > 1:
                print(
                    "You must specify a profile name if multiple profiles "
                    "appear in the JSON config file"
                )
                raise ValueError
            else:
                profile = config.keys()[0]

        return config[profile]

    @staticmethod
    def load_chromsizes(path):
        """
        Loads a fasta file and returns a chromsizes dict.

        Parameters
        ----------
        path : str
            Path to the FASTA file to load.

        Returns
        -------
        chromsizes : dict
            A dictionary of chromosomes sizes format {"chrom": size}


        """
        records = SeqIO.parse(path, format="fasta")
        return {rec.id: len(str(rec.seq)) for rec in records}

    def generate_sv(self):
        """
        Generates random structural variations, based on the parameters loaded
        from the instance's config file.
        # NOTE: Currently only implemented for inversions and deletions.

        Returns
        -------
        pandas.DataFrame :
            A dataframe where each row is a SV. columns represent
            sv_type, chrom, start, end.
        """
        # Relative abundance of each event type (placeholder values)
        # rel_abun = {"INV": 8, "DEL": 400, "DUP": 60, "INS": 160, "CNV": 350}
        all_chroms_sv = []
        for chrom, size in self.chromsizes.items():
            n_sv = round(size * self.config["SV_freq"])
            chrom_sv = pd.DataFrame(np.empty((n_sv, 4)))
            chrom_sv.columns = ["sv_type", "chrom", "start", "end"]
            sv_count = 0
            for sv_name, sv_char in self.config["SV_types"].items():
                # multiply proportion of SV type by total SV freq desired to
                # get number of events of this type.
                n_event = round(n_sv * sv_char["prop"])

                # Safeguard rounding errors
                if sv_count + n_event > n_sv:
                    n_event -= (n_event + sv_count) - n_sv

                print("Generating {0} {1}".format(n_event, sv_name))
                for _ in range(n_event):
                    # Start position is random and length is picked from a normal
                    # distribution centered around mean length.
                    start = np.random.randint(size)
                    end = start + abs(
                        np.random.normal(
                            loc=sv_char["mean_size"], scale=sv_char["sd_size"]
                        )
                    )
                    # Make sure the inversion does not go beyond chromosome.
                    end = min(size, end)
                    chrom_sv.iloc[sv_count, :] = (sv_name, chrom, start, end)
                    sv_count += 1
            all_chroms_sv.append(chrom_sv)
        out_sv = pd.concat(all_chroms_sv, axis=0)
        out_sv.start, out_sv.end = (
            out_sv.start.astype(int),
            out_sv.end.astype(int),
        )
        # Randomize rows in SV table to mix the order of different SV types
        out_sv = out_sv.sample(frac=1).reset_index(drop=True)
        self.sv = out_sv

    def edit_genome(self, fasta_out):
        """
        Given a fasta file and a dataframe of structural variants and their
        positions, generate a new genome by applying the input changes.

        Coordinates in self.sv are updated as the genome is modified.

        Parameters
        ----------
        fasta_out : str
            Path where the edited genome will be written in fasta format.
        """
        with open(fasta_out, "w") as fa_out:
            for rec in SeqIO.parse(self.genome_path, format="fasta"):
                mutseq = Seq.MutableSeq(str(rec.seq))
                for row_num in range(self.sv.shape[0]):
                    row = self.sv.iloc[row_num, :]
                    sv_type = row.sv_type
                    chrom, start, end = row.chrom, int(row.start), int(row.end)
                    # NOTE: Only implemented for inversions for now.
                    if sv_type == "INV":
                        # Reverse complement to generate inversion
                        if chrom == rec.id:
                            mutseq[start:end] = Seq.reverse_complement(
                                mutseq[start:end]
                            )
                            # Update coordinates of other SVs in the INV region
                            mid = (end + start) // 2
                            starts = self.sv.eval(
                                "(chrom == @chrom) & (start >= @start) & (start <= @end)"
                            )
                            ends = self.sv.eval(
                                "(chrom == @chrom) & (end >= @start) & (end <= @end)"
                            )
                            self.sv.loc[starts, "start"] = (
                                mid + mid - self.sv.start[starts]
                            )
                            self.sv.loc[ends, "end"] = (
                                mid + mid - self.sv.end[ends]
                            )
                            # Make sure start is always lower than end
                            swap_mask = self.sv.start > self.sv.end
                            self.sv.loc[
                                swap_mask, ["start", "end"]
                            ] = self.sv.loc[swap_mask, ["end", "start"]].values
                    elif sv_type == "DEL":
                        if chrom == rec.id:
                            mutseq = mutseq[:start] + mutseq[end:]
                            # Shift coordinates on the right of DEL region
                            self.sv.loc[
                                (self.sv.chrom == chrom)
                                & (self.sv.start >= start),
                                ["start", "end"],
                            ] -= (end - start)
                            self.sv.start[self.sv.start < 0] = 0
                            self.sv.end[self.sv.end < 0] = 0
                    else:
                        raise NotImplementedError(
                            "SV type not implemented yet."
                        )
                self.sv.start = self.sv.start.astype(int)
                self.sv.end = self.sv.end.astype(int)
                # Discard SV that have been erased by others
                self.sv = self.sv.loc[(self.sv.end - self.sv.start) > 1, :]
                rec = SeqIO.SeqRecord(seq=mutseq, id=rec.id, description="")
                # Trim SV with coordinates > chrom size
                self.sv.end[
                    (self.sv.chrom == chrom) & (self.sv.end >= len(mutseq))
                ] = (len(mutseq) - 1)
                SeqIO.write(rec, fa_out, format="fasta")


def save_sv(sv_df, clr, path):
    """
    Saves the structural variant (SV) table into a text file.
    The order of SVs in that table matches the order in which they were
    introduced in the genome.
    """
    full_sv = sv_df.copy()
    full_sv["coord_start"] = 0
    full_sv["coord_end"] = 0
    for i in range(full_sv.shape[0]):
        chrom, start, end = full_sv.loc[i, ["chrom", "start", "end"]]
        full_sv.loc[i, ["coord_start", "coord_end"]] = clr.extent(
            f"{chrom}:{start}-{end}"
        )
    full_sv.to_csv(path, sep="\t", index=False)


def pos_to_coord(clr, sv_df):
    """
    Converts start - end genomic positions from structural variations to breakpoints
    in matrix coordinates.

    Parameters
    ----------
    clr : cooler.Cooler
        The cooler object containing Hi-C data
    sv_df : pandas.DataFrame
        A dataframe containg the type and genomic start-end coordinates of
        strucural variations as given by generate_sv().

    Returns
    -------
    breakpoints : numpy.array of int
        A N x 2 numpy array of numeric values representing X, Y coordinates of structural
        variations breakpoints in the matrix.
    labels : numpy.array of str
        An N X 1 array of labels corresponding to SV type.
    """
    # Get coordinates to match binning
    res = clr.binsize
    sv_df.start = (sv_df.start // res) * res
    sv_df.end = (sv_df.end // res) * res
    # Put start and end in the same column, 1 row / breakpoint
    s_df = sv_df.loc[:, ["sv_type", "chrom", "start"]]
    s_df.rename(index=str, columns={"start": "pos"}, inplace=True)
    e_df = sv_df.loc[:, ["sv_type", "chrom", "end"]]
    e_df.rename(index=str, columns={"end": "pos"}, inplace=True)
    sv_df = pd.concat([s_df, e_df]).reset_index(drop=True)
    # Assign matrix coordinate (fragment index) to each breakpoint
    bins = clr.bins()[:]
    bins["coord"] = bins.index
    sv_frags = sv_df.merge(
        bins,
        left_on=["chrom", "pos"],
        right_on=["chrom", "start"],
        how="left",
    )
    breakpoints = np.vstack([sv_frags.coord, sv_frags.coord]).T
    breakpoints.astype(int)
    labels = np.array(sv_frags.sv_type.tolist())
    return breakpoints, labels


def subset_mat(clr, coords, labels, win_size, prop_negative=0.5):
    """
    Samples evenly sized windows from a matrix. Windows are centered around
    input coordinates. Windows and their associated labels are returned. A number
    of random negative windows (without sv) will be added to the output so that
    there are prop_negative negative negative windows in the output.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object containing Hi-C data.
    coords : numpy.ndarray of ints
        Pairs of coordinates for which subsets should be generated. A window
        centered around each of these coordinates will be sampled. Dimensions
        are [N, 2].
    labels : numpy.ndarray of ints
    win_size : int
        Size of windows to sample from the matrix.
    prop_negative : float
        The proportion of windows without SVs desired. If set to 0.5, when given
        a list of 23 SV, the function will output 46 observations (windows); 23
        without SV (picked randomly in the matrix) and 23 with SV.

    Returns
    -------
    x : numpy.ndarray of floats
        The 3D feature vector to use as input in a keras model.
        Dimensions are [N, win_size, win_size].
    y : numpy.array of ints
        The 1D label vector of N values to use as prediction in a keras model.
    """
    h, w = clr.shape
    i_w = int(h - win_size // 2)
    j_w = int(w - win_size // 2)
    sv_to_int = {"INV": 1, "DEL": 2, "INS": 3}
    # Only keep coords far enough from borders of the matrix
    valid_coords = np.where(
        (coords[:, 0] > int(win_size / 2))
        & (coords[:, 1] > int(win_size / 2))
        & (coords[:, 0] < i_w)
        & (coords[:, 1] < j_w)
    )[0]
    coords = coords[valid_coords, :]
    labels = labels[valid_coords]
    # Number of windows to generate (including negative windows)
    n_windows = int(coords.shape[0] // (1 - prop_negative))
    x = np.zeros((n_windows, win_size, win_size), dtype=np.float64)
    y = np.zeros(n_windows, dtype=np.int64)
    if win_size >= min(h, w):
        print("Window size must be smaller than the Hi-C matrix.")
    halfw = win_size // 2
    # Getting SV windows
    coords = coords.astype(int)
    for i in range(coords.shape[0]):
        c = coords[i, :]
        try:
            win = clr.matrix(sparse=False, balance=False)[
                (c[0] - halfw) : (c[0] + halfw),
                (c[1] - halfw) : (c[1] + halfw),
            ]
        except TypeError:
            breakpoint()
        x[i, :, :] = win
        y[i] = sv_to_int[labels[i]]
    # Getting negative windows
    neg_coords = set()
    for i in range(coords.shape[0], n_windows):
        tries = 0
        c = np.random.randint(win_size // 2, i_w)
        # this coordinate must not exist already
        while (c in coords[:, 0]) or (c in neg_coords):
            print("{} is already used. Trying another position...".format(c))
            # If unable to find new coords, just return output until here
            if tries > 100:
                return x[:i, :, :], y[:i]
            neg_coords.add(c)
            c = np.random.randint(win_size // 2, i_w)
            tries += 1
        win = clr.matrix(sparse=False, balance=False)[
            (c - halfw) : (c + halfw), (c - halfw) : (c + halfw)
        ]
        x[i, :, :] = win
        y[i] = 0
        x = x.astype(int)
    return x, y


def matrix_diag_chunks(matrix, size=128, stride=1):
    """Given an input matrix, yield a generator of chunks along the diagonal"""
    m, n = matrix.shape
    if m != n:
        raise ValueError("Input matrix must be square.")
    for i in range(0, matrix.shape - size, stride):
        start = i
        end = start + size
        chunk = matrix[start:end, start:end]
        yield chunk
