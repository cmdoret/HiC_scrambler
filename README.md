# Hi-C scrambled maps generator

> This is a WIP. Here is the state of the different features:

* [x] Boilerplate for editing genomes and generating matrices.
* [x] Storing SV positions and windows.
* [X] Storing pairs of whole maps before and after scrambling
* [ ] Implementing all SV types (only inversions and deletions for now)
* [ ] Generating features from [BAM](https://samtools.github.io/hts-specs/SAMv1.pdf) alignments

This repo contains a program to generate scrambled Hi-C maps. The program starts from an input genome and Hi-C library (reads) and introduces structural variants into the genome. Structural variants (SV) are large scale alteration to the sequence including:

* Deletion: Chunk of sequence removed
* Insertion: New chunk of sequence introduced
* Inversion: Chunk of sequence flipped
* Translocation: Chunk of sequence moved from one place to another
* Duplication: Chunk of sequence copied to a different position.

These alterations can be happen sequentially and be superimposed on each other, which result in "complex events".

The simplest approach to generating scrambled maps would be to directly reorder rows / columns of the matrix, but this would not accurately replicate the artifacts visible in actual SV due to read alignments.

# Setup

To install python dependencies, you can use the requirements.txt file as follows:

```bash
pip install --user -r requirements.txt
```

To setup install the project as a local package, run:

```bash
pip install --user -e .
```

## Usage

The pipeline requires a genome (fasta format) and a Hi-C library (fastq format). 

A json configuration file is provided to define profiles. These profiles dictate the type of SV to generate and their properties (size, frequency, ...).

## Output

The pipeline will generate an output directory containing multiple files.
Each run will run on a random subset of the input genome. It will have its own subdirectory containing the matrix before and after scrambling, as well as the list of SV applied and zoom on the concerned regions.

The root output directory will contain the following files combining all runs:
* `x.npy`: A 3D numpy array (npy) file containing windows around each SV as well as windows around random positions without SV (50/50).
* `y.npy`: A 1D numpy array containing labels corresponding to the windows with the following encoding: 0=no SV, 1=INV, 2=DEL, 3=INS.
* `truth.npy`: A 3D numpy array containing the matrix of each run's random region before scrambling.
* `scrambled.npy`: Same, but after scrambling. Each map is added a bottom and right 0-padding to retain the same dimensions despite deletions.

## Test dataset

Hi-C reads from a hybrid S.cerevisiae - S. paradoxus library. Reads are mapped to chromosome 4 using bowtie2 with very-fast-local preset and extracted all reads mapping to the chromosome.

The matrices were then generated using those reads. The "original" matrix does not contain any structural variation.

All matrices are generated with hicstuff using parameters:
`hicstuff pipeline -t 12 -P original -e 1000 -f genome.fa -m aligned_for.fq.gz aligned_rev.fq.gz -o output`
