# Script used to coordinate all steps to generate input for the model
# Datasets generated consist of an NxWxW array of features and an N array
# of labels
import input_utils as iu
import os
from hicstuff import pipeline

TMP_DIR = "data/tmp"
ORIG_GENOME = "data/genome.fa"
ORIG_CHROMSIZE = "data/input/original.chr.tsv"
OUT_DIR = "data/input/training/"
N_RUNS = 1
### Make edited genomes
for i in range(N_RUNS):
    RUN_DIR = OUT_DIR.rstrip("/") + "_" + i
    os.makedir(RUN_DIR)
    # Generate random structural variations and apply them to the genome
    struct_vars = iu.generate_sv(ORIG_CHROMSIZE)
    iu.edit_genome(ORIG_GENOME, os.path.join(RUN_DIR, "mod_genome.fa", sv))
    # Generate contact map using the edited genome
    pl = pipeline(
        [
            "-F",
            "-e",
            "DpnII",
            "-t",
            "12",
            "-m",
            "-o",
            RUN_DIR,
            "-f",
            ORIG_GENOME,
            "data/reads_for.fastq",
            "data/reads_rev.fastq",
            "-P",
            RUN,
            "-T",
            TMP_DIR,
        ]
    )
    pl.execute()

    X, Y = iu.subset_mat(run_mat, struct_vars, labels)
