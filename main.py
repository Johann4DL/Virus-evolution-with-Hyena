from torch.utils.data import DataLoader
from typing import List, Optional, Tuple 

import torch
import torch.nn as nn
import dataclasses
import math
import numpy as np
import regex
import wandb
from tqdm.auto import tqdm

from hyena_simp import Config, HyenaConfig, AuthenticHyenaBlock, FastaModel
from utils import CovGenomes, read_fasta, count_parameters
from config import hyena_config

CONTEXT_LENGTH = max([len(x) for x in source + target])


# load data and preprocess

source_file_name = './data/British_Columbia/BC_Jan_2023/gisaid_auspice_input_hcov-19_2024_02_01_22/1706826790587.sequences.fasta'
target_file_name = './data/British_Columbia/BC_Feb_2023/gisaid_auspice_input_hcov-19_2024_02_01_22/1706826876642.sequences.fasta'


source = read_fasta(source_file_name)
target = read_fasta(target_file_name)

temp = source + target
min_length = min([len(x) for x in temp])
max_length = max([len(x) for x in temp])

print('Min and max length of genome sequences before padding:\nMin: ', min_length,'\nMax: ', max_length)

# apply 'N' padding to the sequences
max_length = max([len(x) for x in source + target])

for i in range(len(source)):
    source[i] = source[i] + 'N' * (max_length - len(source[i]))
    
for i in range(len(target)):
    target[i] = target[i] + 'N' * (max_length - len(target[i]))


min_length = min([len(x) for x in source + target])
max_length = max([len(x) for x in source + target])

print('Min and max length of genome sequences after padding:\nMin: ', min_length,'\nMax: ', max_length)


# Tokenize

# unique chars in source_genomes
chars = set()

for genome in source + target:
    for char in genome:
        chars.add(char)
vocabulary = list(chars)

tok2id = {ch: i for i, ch in enumerate(vocabulary)}
id2tok = {i: ch for i, ch in enumerate(vocabulary)}
# print(tok2id)

encode = lambda s: [tok2id[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([id2tok[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits

n = int(0.9*len(source)) # first 90% will be train, rest val
train_source = source[:n]
val_source = source[n:]

n = int(0.9*len(target))
train_target = target[:n]
val_target = target[n:]


# Datasets

train_ds = CovGenomes(train_source, train_target)
val_ds = CovGenomes(val_source, val_target)

# Dataloader
loader = DataLoader(train_ds, batch_size=hyena_config.batch_size, shuffle=True, num_workers=10)
val_loader = DataLoader(val_ds, batch_size=hyena_config.batch_size, shuffle=True, num_workers=10)

model = FastaModel(hyena_config, AuthenticHyenaBlock)
m = model.to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f'The model has {count_parameters(model):,} trainable parameters')


# training loop

for iter in tqdm(range(hyena_config.epochs)):
    for source, target in loader:
        source = source.to('cuda')
        target = target.to('cuda')
        
        logits, y = model(source, target)
        loss = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), target
                    )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    wandb.log({"loss": loss.item(),
                "epoch": iter,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "batch_size": hyena_config.batch_size,
                "num_workers": hyena_config.num_workers,})


# save model