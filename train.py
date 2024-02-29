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
from utils import read_fasta, count_parameters
from config import hyena_config


# load data

file_name = './data/British_Columbia/BC_Jan_2023/gisaid_auspice_input_hcov-19_2024_02_01_22/1706826790587.sequences.fasta'

data = read_fasta(file_name)

CONTEXT_LENGTH = max([len(x) for x in data])

min_length = min([len(x) for x in data])
max_length = max([len(x) for x in data])

print('Min and max length of genome sequences before padding:\nMin: ', min_length,'\nMax: ', max_length)

# apply 'N' padding to the sequences
max_length = max([len(x) for x in data])

for i in range(len(data)):
    data[i] = data[i] + 'N' * (max_length - len(data[i]))

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
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


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

for iter in tqdm(range(hyena_config.epochs)):
    for source in loader:
        source = source.to('cuda')
        
        logits = model(source)
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), source
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(iter)
        
    # wandb.log({"loss": loss.item(),
    #             "epoch": iter+1,
    #             "learning_rate": optimizer.param_groups[0]['lr'],
    #             "batch_size": hyena_config.batch_size,})


# save model
torch.save(model, 'models/model_scripted.pt')
# torch.save(model.state_dict(), 'model_scripted_SD.pt')