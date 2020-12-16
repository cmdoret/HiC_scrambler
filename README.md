# Hi-C scrambled maps generator

> This is a WIP. Here is the state of the different features:

* [x] Boilerplate for editing genomes and generating matrices.
* [x] Storing SV positions and windows.
* [ ] Implementing all SV types (only inversions for now)

This repo contains a program to generate scrambled Hi-C maps. The program starts from an input genome and Hi-C library (reads) and introduces structural variants into the genome. Structural variants (SV) are large scale alteration to the sequence including:

* Deletion: Chunk of sequence removed
* Insertion: New chunk of sequence introduced
* Inversion: Chunk of sequence flipped
* Translocation: Chunk of sequence moved from one place to another
* Duplication: Chunk of sequence copied to a different position.

These alterations can be happen sequentially and be superimposed on each other, which result in "complex events".

The simplest approach to generating scrambled maps would be to directly reorder rows / columns of the matrix, but this would not accurately replicate the artifacts visible in actual SV due to read alignments.

## Usage

The pipeline requires a genome (fasta format) and an optional Hi-C library (fastq format). If no library is supplied, a synthetic library will be generated from the genome to generate a matrix with a simple gradient without particular 3D structure.

A YAML configuration file is provided to define profiles. These profiles dictate the type of SV to generate and their properties (size, frequency, ...).

## Output

The pipeline will generate an output directory containing multiple files.
Each run will have its own subdirectory containing the cool file of the scrambled map and the list of SV applied.

The root output directory will contain 2 files combining all runs:
* The features: A 3D numpy array (npy) file containing windows around each SV as well as windows around random positions without SV (50/50).
* The labels: A 1D numpy array containing labels corresponding to the windows with the following encoding: 0=no SV, 1=INV, 2=DEL, 3=INS

file in cool format for the original matrix and one cool file per scramble run. For each scramble run, it will generate a matching text file containing the list of structural variants.

## Test dataset

Hi-C reads from a hybrid S.cerevisiae - S. paradoxus library. Reads are mapped to chromosome 4 using bowtie2 with very-fast-local preset and extracted all reads mapping to the chromosome.

The matrices were then generated using those reads. The "original" matrix does not contain any structural variation.

All matrices are generated with hicstuff using parameters:
`hicstuff pipeline -t 12 -P original -e 1000 -f genome.fa -m aligned_for.fq.gz aligned_rev.fq.gz -o output`
