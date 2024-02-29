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
from utils import read_fasta, count_parameters, Embeddings_DS
from config import hyena_config


# load data

path = './data/British_Columbia/BC_Jan_2023/gisaid_auspice_input_hcov-19_2024_02_01_22/1706826790587.sequences.fasta'
data = read_fasta(path)

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

# unique chars in source_genomes
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

n = int(0.9*len(source)) # first 90% will be train, rest val
train_source = source[:n]
val_source = source[n:]


# Datasets

train_ds = Embeddings_DS(train_source)
val_ds = Embeddings_DS(val_source)

# Dataloader
loader = DataLoader(train_ds, batch_size=hyena_config.batch_size, shuffle=True, num_workers=10)
val_loader = DataLoader(val_ds, batch_size=hyena_config.batch_size, shuffle=True, num_workers=10)

# Optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastaModel(hyena_config, AuthenticHyenaBlock)
model = nn.DataParallel(model)
model.to(device)
# m = model.to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


print(f'The model has {count_parameters(model):,} trainable parameters')

# wandb
#start a new wandb run to track this script
# wandb.login(key=026200c66c6b5dd3cc0d7961b457f0ad939a1add)

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
        model.train()
        source = source.to('cuda')
        
        logits = model(source)
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), source
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    # wandb.log({"loss": loss.item(),
    #             "epoch": iter,
    #             "learning_rate": optimizer.param_groups[0]['lr'],
    #             "batch_size": hyena_config.batch_size,
    #             "num_workers": hyena_config.num_workers,})
    # val_loss = 0
    # for batch in val_loader:
    #     model.eval()
    #     batch = batch.to(device)
    #     logits = model(batch)
    #     val_loss += torch.nn.functional.cross_entropy(
    #         logits.transpose(1, 2), source
    #     )



# save model
# PATH = 'models/model_weights'
# torch.save(the_model.state_dict(), PATH)

torch.save(model, 'models/model_scripted.pt')