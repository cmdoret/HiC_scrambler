# Script used to generate training data for the model
# Datasets generated consist of an NxWxW array of features and an N array
# of labels
from hic_scrambler import input_utils as iu
from hic_scrambler import genome_utils as gu
import os
from os.path import join
import cooler
import hicstuff.pipeline as hpi
import numpy as np
import pathlib
import click

CONFIG_PATH = join(os.path.dirname(__file__), "config", "template.json")


@click.command()
@click.option(
    "--reads1",
    "-1",
    default=None,
    help="Forward Hi-C reads",
)
@click.option(
    "--reads2",
    "-2",
    default=None,
    help="Reverse Hi-C reads",
)
@click.option(
    "--binsize",
    "-b",
    default=2000,
    show_default=True,
    help="The resolution of matrices to generate, in basepair",
)
@click.option(
    "--nruns",
    "-n",
    default=1,
    help="The number of scramble runs to generate",
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
    0. Select a random region of a random chromosome
    1. Edit the selected region to add SV
    2. Realign the reads and generate a (cool) Hi-C map from the scrambled genome
    3. Extract windows around each SV as well as random (negative) windows
    4. Store the windows and associated labels into npy files.
    5. Store the whole matrix before and after SV.
    """
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    if reads1 is None or reads2 is None:
        raise NotImplementedError("Reads generation not implemented yet.")
    # Generate an initial contact map for the original genome
    hpi.full_pipeline(
        fasta,
        reads1,
        reads2,
        aligner="bowtie2",
        tmp_dir=tmpdir,
        out_dir=outdir,
        prefix="original",
        threads=8,
        enzyme=binsize,
        mat_fmt="cool",
    )
    clr_ori = cooler.Cooler(join(outdir, "original.cool"))
    slice_bins = 512
    # Make edited genomes. Each edited genome will start from a subset of the
    # original (full) genome.
    for i in range(nruns):
        rundir = join(outdir, f"RUN_{i}")

        # Subset genome: Pick a random chromosome and slice to generate a matrix of 1000 x 1000
        os.makedirs(rundir, exist_ok=True)
        sub_fasta = join(rundir, "genome.fa")
        slice_region = gu.slice_genome(
            fasta, sub_fasta, slice_size=slice_bins * binsize
        )

        # Save map corresponding to the slice region (before SV)
        mat_ori = clr_ori.matrix(sparse=False, balance=False).fetch(slice_region)
        np.save(join(rundir, "truth.npy"), mat_ori)

        # Generate random structural variations and apply them to the genome
        mixer = gu.GenomeMixer(sub_fasta, CONFIG_PATH, "Debug")
        mixer.generate_sv()
        mod_genome = join(rundir, "mod_genome.fa")
        mixer.save_edited_genome(mod_genome)
        # Generate contact map using the edited genome
        hpi.full_pipeline(
            mod_genome,
            reads1,
            reads2,
            aligner="bowtie2",
            tmp_dir=tmpdir,
            out_dir=rundir,
            prefix="scrambled",
            threads=8,
            enzyme=binsize,
            mat_fmt="cool",
        )
        # Extract window around each SV and as many random windows
        clr_mod = cooler.Cooler(join(rundir, "scrambled.cool"))
        breakpoints, labels = gu.pos_to_coord(clr_mod, mixer.sv)
        X, Y = gu.subset_mat(
            clr_mod, breakpoints, labels, win_size=128, prop_negative=0.5
        )
        # Save whole slice map (after SV)
        np.save(
            join(rundir, f"scrambled.npy"),
            clr_mod.matrix(sparse=False, balance=False)[:],
        )

        # Save all corresponding Hi-C windows and associated label (SV type) to a file
        np.save(join(rundir, "x.npy"), X)
        np.save(join(rundir, "y.npy"), Y)
        # Save list of SVs coordinates
        gu.save_sv(mixer.sv, clr_mod, join(rundir, "breakpoints.tsv"))

    # For convenience, also generate a file with the windows, labels and
    # matrices from all combined runs

    # Helper function to concatenate piles of images
    conc = lambda base: np.concatenate([np.load(base.format(i)) for i in range(nruns)])
    feats = conc(join(outdir, "RUN_{}", "x.npy"))
    labs = conc(join(outdir, "RUN_{}", "y.npy"))
    np.save(join(outdir, "x.npy"), feats)
    np.save(join(outdir, "y.npy"), labs)

    # Helper function to pad individual images to the same size and stack them
    # NOTE: Padding is required because we introduced deletions.
    stack = lambda base: np.dstack(
        [iu.pad_matrix(np.load(base.format(i)), slice_bins + 1) for i in range(nruns)]
    ).transpose(2, 0, 1)

    oris = stack(join(outdir, "RUN_{}", "truth.npy"))
    mods = stack(join(outdir, "RUN_{}", "scrambled.npy"))
    np.save(join(outdir, "truth.npy"), oris)
    np.save(join(outdir, "scrambled.npy"), mods)


if __name__ == "__main__":
    run_scrambles()  # pylint: disable=no-value-for-parameter
