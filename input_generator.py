# Script used to generate input for the model
# Datasets generated consist of an NxWxW array of features and an N array
# of labels
import input_utils as iu
import os
from os.path import join
from hicstuff.commands import Pipeline
import hicstuff.io as hio
import numpy as np
import pandas as pd
import pathlib


BINSIZE = 2000
TMP_DIR = "data/tmp"
ORIG_GENOME = "data/genome.fa"
ORIG_CHROMSIZE = "data/input/original.chr.tsv"
OUT_DIR = "data/input/training/"
N_RUNS = 1

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

chromsizes = iu.load_chromsizes(ORIG_CHROMSIZE)
# Make edited genomes
for i in range(N_RUNS):
    RUN_DIR = OUT_DIR.rstrip("/") + "_" + str(i)
    os.makedirs(RUN_DIR, exist_ok=True)
    # Generate random structural variations and apply them to the genome
    struct_vars = iu.generate_sv(chromsizes, freq=10e-7)
    mod_genome = join(RUN_DIR, "mod_genome.fa")
    iu.edit_genome(ORIG_GENOME, mod_genome, struct_vars)
    # Generate contact map using the edited genome
    pl = Pipeline(
        [
            "-e",
            str(BINSIZE),
            "-t",
            "12",
            "-m",
            "-o",
            RUN_DIR,
            "-f",
            mod_genome,
            "data/aligned_for.fq",
            "data/aligned_rev.fq",
            "-P",
            "RUN_" + str(i),
            "-T",
            TMP_DIR,
        ],
        {},
    )
    pl.execute()
    mat_path = join(RUN_DIR, "RUN_" + str(i) + ".mat.tsv")
    frags = pd.read_csv(join(RUN_DIR, "RUN_" + str(i) + ".frag.tsv"), delimiter="\t")
    run_mat = hio.load_sparse_matrix(mat_path)
    coords, labels = iu.pos_to_coord(struct_vars, frags, BINSIZE)
    X, Y = iu.subset_mat(run_mat, coords.astype(int), labels)
    np.save(join(OUT_DIR, "RUN_" + str(i) + ".x.npy"), X)
    np.save(join(OUT_DIR, "RUN_" + str(i) + ".y.npy"), Y)
