import torch
import torch.nn as nn
import numpy as np
from typing import List
import regex
from torch.utils.data import DataLoader

def read_fasta(file_name: str) -> List[str]:
    file = open(file_name, 'r') # identifier line starts with >
    sequence = ''
    genomes = []

    for line in file.readlines():
        if line[0] == '>':
            genomes.append(sequence)
            sequence = ''
        else:
            # remove new line character
            temp = line[:-1]
            sequence = sequence + temp

    output = []
    for genome in genomes:
        # remove all non ACGT characters
        genome = regex.sub('[^ACGTN]', 'N', genome)
        output.append(genome)

    # remove seqeuences that are shorter than 29000
    genomes = [x for x in genomes if len(x) > 29000]

    # add start and end tokens
    genomes = ['^' + x + '$' for x in genomes]

    return genomes

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

