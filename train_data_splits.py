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
import os

from hyena_simp import Config, HyenaConfig, AuthenticHyenaBlock, FastaModel
from utils import count_parameters, read, train
from config import hyena_config


# get number of files in a directory
def get_num_files(directory: str) -> int:
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path
        .join(directory, name))])

path = './data/data_splits/'

# get number of files in the directory
num_files = get_num_files(path)
print(f'Number of files in the directory: {num_files}')


CONTEXT_LENGTH = 30000

def preprocess(data, context_length):
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
    return data


for file in os.listdir(path):
    if file.endswith('.fasta'):
        path = './data/data_splits/{file}'.format(file=file)
        print(f'Processing file: {file}')
        data = read(path)
        data = preprocess(data, CONTEXT_LENGTH)

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

        print('Number of genome sequences: ',len(data))
        print('Number of unique characters: ', len(vocabulary))

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
        
        # Train and test splits

        n = int(0.9*len(data)) # first 90% will be train, rest val
        train_data = data[:n]
        val_data = data[n:]

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
     

        # training loop

        train(model, loader, val_loader, optimizer, hyena_config)  

        # save model
        torch.save(model.state_dict(), 'models/model_state_dict_{}.pt'.format(file))

        # write val data to file
        with open('val_data.txt', 'w') as f:
            for item in val_data:
                f.write("%s\n" % item)

        # append val_data.txt to all_val_data.txt
        with open('all_val_data.txt', 'a') as f:
            for item in val_data:
                f.write("%s\n" % item)