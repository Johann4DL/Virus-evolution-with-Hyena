import torch
import torch.nn as nn
import numpy as np
from typing import List
import regex
import glob
from tqdm.auto import tqdm
import wandb


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read(file_name: str) -> List[str]:
    file = open(file_name, 'r') # identifier line starts with >
    sequence = ''
    genomes = []
    state = False

    for line in file.readlines():
        if line[0] == '>' and state == False:
            sequence = '>'
            state = True
        elif line[0] != '>' and state == True:
            # remove new line character
            temp = line[:-1]
            sequence = sequence + temp
            # genomes.append(sequence)
            # state = False
        elif line[0] == '>' and state == True:
            genomes.append(sequence)
            sequence = '>'  

    # remove seqeuences that are shorter than 29000
    genomes = [x for x in genomes if len(x) > 29000] # and len(x) <= 29800]

    return genomes

def load_data(path):
    '''
    Load the data from a fasta file and return a list of sequences
    '''
    all_files = glob.glob(path + "/*.fasta")
    genomes = []
    for filename in all_files:
        temp = read(filename)
        genomes = genomes + temp
    return genomes

def train(model, loader, val_loader, optimizer, hyena_config):

    for iter in tqdm(range(hyena_config.epochs)):
        train_loss_ = []
        model.train()
        for source in loader:
            source = source.to('cuda')
            
            logits, genome_embedding = model(source)
            loss = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), source
            )
            train_loss_.append(loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        tot_train_loss = sum(train_loss_) / len(train_loss_)
        if (iter+1) % 5 == 0:
            print(f'Training loss: {tot_train_loss}')

            # wandb.log({"train_loss": tot_train_loss.item(),
            #                 "epoch": iter+1,
            #                 "learning_rate": optimizer.param_groups[0]['lr'],})
            val_loss_ = []
            model.eval()
            for batch in val_loader:
                batch = batch.to('cuda')
                logits, genome_embedding = model(batch)
                loss = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), batch
                )
                val_loss_.append(loss.item())

            # accumulate val loss
            tot_val_loss = sum(val_loss_) / len(val_loss_) 
            print(f'Validation loss: {tot_val_loss}')
        
            # wandb.log({"validation_loss": tot_val_loss.item(),
            #             "epoch": iter+1,
            #             "learning_rate": optimizer.param_groups[0]['lr'],
            #             "batch_size": hyena_config.batch_size,})
