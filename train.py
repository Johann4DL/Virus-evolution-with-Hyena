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
split = 4500
#path = './data/all_genomes.fasta'
path = './data/data_splits/2023_2024_genomes_{}.fasta'.format(split)

data = read(path)
# selsct the first 2250 sequences
#data = data[0:2250]
print('Number of genome sequences: ',len(data))

# preprocessing
# CONTEXT_LENGTH = max([len(x) for x in data])
CONTEXT_LENGTH = 30000

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

for genome in data:
    for char in genome:
        chars.add(char)
vocabulary = list(chars)


tok2id = {ch: i for i, ch in enumerate(vocabulary)}
id2tok = {i: ch for i, ch in enumerate(vocabulary)}

encode = lambda s: [tok2id[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([id2tok[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

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

# Datasets

train_ds = Embeddings_DS(train_data)
val_ds = Embeddings_DS(val_data)

# Dataloader
loader = DataLoader(train_ds, batch_size=hyena_config.batch_size, shuffle=True, num_workers=10)
val_loader = DataLoader(val_ds, batch_size=hyena_config.batch_size, shuffle=True, num_workers=10)

model = FastaModel(hyena_config, AuthenticHyenaBlock)
m = model.to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


print(f'The model has {count_parameters(model):,} trainable parameters')

# wandb
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Hyena_Covid",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": hyena_config.learning_rate,
#     "epochs": hyena_config.epochs,
#     }
# )

# training loop

train(model, loader, val_loader, optimizer, hyena_config)  # 1 epoch ~ 180 seconds

# save model
torch.save(model.state_dict(), 'models/model_state_dict_{}.pt'.format(split))

# write val data to file
with open('val_data.txt', 'w') as f:
    for item in val_data:
        f.write("%s\n" % item)

# append val_data.txt to all_val_data.txt
with open('all_val_data.txt', 'a') as f:
    for item in val_data:
        f.write("%s\n" % item)
# torch.save(model, 'models/model_1.pt')