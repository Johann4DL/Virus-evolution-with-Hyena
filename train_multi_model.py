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
from utils import count_parameters, read, train
from config import hyena_config

# load data

path = './data/2023_1_genomes.fasta'

data = read(path)
print('Number of genome sequences: ',len(data))

# preprocessing
CONTEXT_LENGTH = 30000

# cut sequences to CONTEXT_LENGTH

for i in range(len(data)):
    if len(data[i]) >= CONTEXT_LENGTH:
        data[i] = data[i][:CONTEXT_LENGTH]

# cut sequences to CONTEXT_LENGTH
for i in range(len(data)):
    if len(data[i]) >= CONTEXT_LENGTH:
        data[i] = data[i][:CONTEXT_LENGTH]
# apply 'P' padding to the sequences
for i in range(len(data)):
    data[i] = data[i] + 'P' * (CONTEXT_LENGTH - len(data[i]))


min_length = min([len(x) for x in data])
max_length = max([len(x) for x in data])

print('Min and max length of genome sequences after padding:\nMin: ', min_length,'\nMax: ', max_length)

# Tokenize

chars = set()
print('Number of genome sequences: ',len(data))
for genome in data:
    for char in genome:
        chars.add(char)
vocabulary = list(chars)
# print('Vocabulary:', vocabulary)


tok2id = {ch: i for i, ch in enumerate(vocabulary)}
id2tok = {i: ch for i, ch in enumerate(vocabulary)}

encode = lambda s: [tok2id[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([id2tok[i] for i in l]) # decoder: take a list of integers, output a string


# split data into chunks of 2250
data = [data[i:i + 2250] for i in range(0, len(data), 2250)]
print('Number of chunks: ', len(data))

class Embeddings_DS(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        idx = np.random.randint(len(self.data))
        data = self.data[idx]
        data = torch.tensor(encode(data))

        return data

# train model
for i in range(len(data)):
    # train test split
    data_chunk = data[i]
    print('Chunk number: ', i)
    print('Number of genome sequences in chunk: ', len(data_chunk))
    n = int(0.9*len(data_chunk)) # first 90% will be train, rest val
    train_data = data_chunk[:n]
    val_data = data_chunk[n:]

    # write val data to file
    with open('val_data.txt', 'w') as f:
        for item in val_data:
            f.write("%s\n" % item)

    # append val_data.txt to all_val_data.txt
    with open('all_val_data.txt', 'a') as f:
        for item in val_data:
            f.write("%s\n" % item)
    train_ds = Embeddings_DS(train_data)
    val_ds = Embeddings_DS(val_data)

    # Dataloader
    loader = DataLoader(train_ds, batch_size=hyena_config.batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_ds, batch_size=hyena_config.batch_size, shuffle=True, num_workers=10)

    model = FastaModel(hyena_config, AuthenticHyenaBlock)
    m = model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # training loop

    train(model, loader, val_loader, optimizer, hyena_config)  # 1 epoch ~ 180 seconds

    # save model
    torch.save(model.state_dict(), 'models/model_state_dict_{}.pt'.format(i))
    # torch.save(model, 'models/model_{}.pt'.format(i))

    