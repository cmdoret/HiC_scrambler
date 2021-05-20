# Functions which are used to compute the evolution of GC%.
import numpy as np
from Bio import SeqIO

def load_seq(path : str, chrom_id : str, ind_beg : int, ind_end : int):
    """
    Return the sequence of a chromosome between two indexes.
  
    Parameters:
    ----------
    path : str 
        Path where the sequences are.

    chrom_id : str
        Id of the chromosome we want.

    ind_beg : int
        Index of the begin of the part of the sequence we want. 
        
    ind_end : int
        Index of the end of the part of the sequence we want. 
    """

    records = SeqIO.parse(path, format="fasta")
    

    for rec in records:
        if rec.id == chrom_id:
            seq_to_return = str(rec.seq)[ind_beg:ind_end]
            break
    return seq_to_return

def percent_GC(seq : str):
    """
    Return the evolution of the GC% for a sequence.
  
    Parameters:
    ----------
    path : str 
        Sequence of ACTG.
    """

    number_G = seq.count('G')
    number_C = seq.count('C')

    percents_GC = (number_G + number_C)/len(seq)

    return percents_GC