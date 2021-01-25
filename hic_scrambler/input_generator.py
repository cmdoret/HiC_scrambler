# Script used to generate training data for the model
# Datasets generated consist of an NxWxW array of features and an N array
# of labels
import input_utils as iu
import os
from os.path import join
import cooler
import hicstuff.pipeline as hpi
import hicstuff.io as hio
import numpy as np
import shutil as su
import pandas as pd
import pathlib
import click

CONFIG_PATH = join(os.path.dirname(__file__), "config", "template.json")


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
    if reads1 is None or reads2 is None:
        raise NotImplementedError("Reads generation not implemented yet.")
    # Make edited genomes
    for i in range(nruns):
        rundir = join(outdir, f"RUN_{i}")
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
            aligner="bowtie2",
            tmp_dir=tmpdir,
            out_dir=rundir,
            prefix=f"RUN_{i}",
            threads=8,
            enzyme=binsize,
            mat_fmt="cool",
        )
        # Extract window around each SV and as many random windows
        clr = cooler.Cooler(join(rundir, f"RUN_{i}.cool"))
        breakpoints, labels = iu.pos_to_coord(clr, mixer.sv)
        X, Y = iu.subset_mat(
            clr, breakpoints, labels, win_size=128, prop_negative=0.5
        )
        # Save all corresponding Hi-C windows and associated label (SV type) to a file
        np.save(join(rundir, "x.npy"), X)
        np.save(join(rundir, "y.npy"), Y)
        # Save list of SVs coordinates
        iu.save_sv(mixer.sv, clr, join(rundir, "breakpoints.tsv"))

    # For convenience, also generate a file with the windows and labels from
    # all combined runs
    feats = np.concatenate(
        [np.load(join(outdir, f"RUN_{i}", "x.npy")) for i in range(nruns)]
    )
    labs = np.concatenate(
        [np.load(join(outdir, f"RUN_{i}", "y.npy")) for i in range(nruns)]
    )
    np.save(join(outdir, "x.npy"), feats)
    np.save(join(outdir, "y.npy"), labs)


if __name__ == "__main__":
    run_scrambles()  # pylint: disable=no-value-for-parameter

