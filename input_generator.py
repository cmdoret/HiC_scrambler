# Script used to generate training data for the model
# Datasets generated consist of an NxWxW array of features and an N array
# of labels
import input_utils as iu
import os
from os.path import join
from hicstuff.commands import Pipeline
import hicstuff.io as hio
import numpy as np
import shutil as su
import pandas as pd
import pathlib


BINSIZE = 2000
TMP_DIR = "data/tmp"
ORIG_GENOME = "data/genome.fa"
CONFIG_PATH = "template.json"
PROFILE = "Debug"
OUT_DIR = join("data/input/training/", PROFILE)
N_RUNS = 5

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

mixer = iu.GenomeMixer(ORIG_GENOME, CONFIG_PATH, "Debug")
# Make edited genomes
for i in range(N_RUNS):
    RUN_DIR = OUT_DIR.rstrip("/") + "_" + str(i)
    os.makedirs(RUN_DIR, exist_ok=True)
    # Generate random structural variations and apply them to the genome
    mixer.generate_sv()
    mod_genome = join(RUN_DIR, "mod_genome.fa")
    mixer.edit_genome(mod_genome)
    # Generate contact map using the edited genome
    pl = Pipeline(
        [
            "-e",
            str(BINSIZE),
            "-t",
            "12",
            "-a",
            "minimap2",
            "-o",
            RUN_DIR,
            "-g",
            mod_genome,
            "data/for.fq",
            "data/rev.fq",
            "-P",
            "RUN_" + str(i),
            "-T",
            TMP_DIR,
        ],
        {},
    )
    pl.execute()
    mat_path = join(RUN_DIR, "RUN_" + str(i) + ".mat.tsv")
    frag_path = join(RUN_DIR, "RUN_" + str(i) + ".frags.tsv")
    frags = pd.read_csv(frag_path, delimiter="\t")
    run_mat = hio.load_sparse_matrix(mat_path)
    slicer = iu.MatrixSlicer(run_mat)
    slicer.pos_to_coord(mixer.sv, frags, BINSIZE)
    X, Y = slicer.subset_mat(win_size=128, prop_negative=0.5)
    np.save(join(OUT_DIR, "RUN_" + str(i) + ".x.npy"), X)
    np.save(join(OUT_DIR, "RUN_" + str(i) + ".y.npy"), Y)
    su.rmtree(RUN_DIR)
