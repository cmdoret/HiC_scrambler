#!/bin/env python3
# Unit tests related to input generation
# cmdoret, 20190302

import numpy as np
import tempfile
from ..hic_scrambler import genome_utils as gu
import pytest
from Bio import SeqIO


def test_edit_genome():
    ...

def test_generate_sv():
    ...

def test_slice_genome():
    in_fa = tempfile.NamedTemporaryFile('w', delete=False)
    out_fa= tempfile.NamedTemporaryFile('w', delete=False)
    out_fa.close
    chroms = {'c1': 1000, 'c2': 500, 'c3': 100}
    for c_name, c_len in chroms.items():
        c_seq = np.random.choice(['A', 'C', 'T', 'G'], size=c_len)
        c_seq = ''.join(c_seq)
        in_fa.write(f'>{c_name}\n')
        in_fa.write(f'{c_seq}\n')
    in_fa.close()
    gu.slice_genome(in_fa.name, out_fa.name, slice_size=20)
    # SeqIO.read will raise an exception if the file contains > 1 record
    genome = SeqIO.read(out_fa.name, 'fasta')
    assert len(genome.seq) == 20