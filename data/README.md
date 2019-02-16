### Test dataset

Hi-C reads from a hybrid S.cerevisiae - S. paradoxus library. Mapped the reads to chromosome 4 using bowtie2 with very-fast-local preset and extracted all reads mapping to the chromosome.

The matrices were then generated using those reads. The "original" matrix does not contain any structural variation.

All matrices are generated with hicstuff using parameters:

`hicstuff pipeline -t 12 -P original -e 1000 -f genome.fa -m aligned_for.fq.gz aligned_rev.fq.gz -o output`
