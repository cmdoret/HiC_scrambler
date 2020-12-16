# Script used to generate training data for the model
# Datasets generated consist of an NxWxW array of features and an N array
# of labels
import input_utils as iu
import os
from os.path import join
import hicstuff.pipeline as hpi
import hicstuff.io as hio
import numpy as np
import pandas as pd
import pathlib
import click

CONFIG_PATH = "template.json"


@click.command()
@click.option(
    "--reads1", "-1", default=None, help="Forward Hi-C reads",
)
@click.option(
    "--reads2", "-2", default=None, help="Reverse Hi-C reads",
)
@click.option(
    "--binsize",
    "-b",
    default=2000,
    show_default=True,
    help="The resolution of matrices to generate, in basepair",
)
@click.option(
    "--nruns", "-n", default=1, help="The number of scramble runs to generate",
)
@click.option(
    "--tmpdir",
    "-t",
    default="./tmp",
    help="Temporary directory to use for the runs.",
)
@click.argument("fasta", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path(exists=False))
def run_scrambles(fasta, outdir, reads1, reads2, binsize, nruns, tmpdir):
    """
    This is the orchestrator function that handles the end-to-end pipeline. For
    each scramble run, it will:
    1. Edit the genome to add SV
    2. Realign the reads and generate a (cool) Hi-C map from the scrambled genome
    3. Extract windows around each SV as well as random (negative) windows
    4. Store the windows and associated labels into npy files.
    """
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    mixer = iu.GenomeMixer(fasta, CONFIG_PATH, "Debug")
    # Make edited genomes
    for i in range(nruns):
        rundir = outdir + f"_{i}"
        os.makedirs(rundir, exist_ok=True)
        # Generate random structural variations and apply them to the genome
        mixer.generate_sv()
        mod_genome = join(rundir, "mod_genome.fa")
        mixer.edit_genome(mod_genome)
        # Generate contact map using the edited genome
        hpi.full_pipeline(
            mod_genome,
            reads1,
            reads2,
            aligner="bwa",
            tmp_dir=tmpdir,
            out_dir=rundir,
            prefix=f"RUN_{i}",
            threads=4,
            enzyme=binsize,
            mat_fmt="cool",
        )
        # Extract window around each SV and as many random windows
        cool_path = join(rundir, f"RUN_{i}.cool")
        slicer = iu.MatrixSlicer(cool_path, mixer.sv)
        slicer.pos_to_coord()
        X, Y = slicer.subset_mat(win_size=128, prop_negative=0.5)
        # Save all those windows and associated label (SV type) to a file
        np.save(join(rundir, "x.npy"), X)
        np.save(join(rundir, "y.npy"), Y)

    # For convenience, also generate a file with the windows and labels from
    # all combined runs
    feats = np.load(join(outdir + "_0", "x.npy"))
    labs = np.load(join(outdir + "_0", "x.npy"))
    for i in range(1, nruns):
        feats = np.append(feats, np.load(join(outdir + f"_{i}", "x.npy")))
        labs = np.append(labs, np.load(join(outdir + f"_{i}", "y.npy")))
    np.save(join(outdir, "x.npy"), feats)
    np.save(join(outdir, "y.npy"), labs)
