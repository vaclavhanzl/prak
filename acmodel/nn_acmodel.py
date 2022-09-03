
# NN acoustic models

#from matrix import *
#import numpy as np
#import torch

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


from prongen.hmm_pron import HMM


from matrix import *



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def get_training_hmms(nn_train_tsv_file, derivatives=2):
    df = pd.read_csv(nn_train_tsv_file, sep="\t", keep_default_na=False)
    hmms = []
    for wav, sentence, targets in list(zip(df.wav.values, df.sentence.values, df.targets.values)):
        hmm = HMM(sentence, wav=wav, derivatives=derivatives)
        hmm.targets = targets
        hmms.append(hmm)
    return hmms


def collect_training_material(hmms):
    b_set = sorted({*"".join([hmm.b for hmm in hmms ])}) # make sorted set of all phone names in the training set
    out_size = len(b_set)
    in_size = hmms[0].mfcc.size(1)
    all_targets = "".join([hmm.targets for hmm in hmms])
    train_len = len(all_targets)
    all_mfcc = torch.cat([hmm.mfcc for hmm in hmms]).double().to(device)
    assert all_mfcc.size()[0]==train_len
    return all_mfcc, all_targets, b_set


class SpeechDataset(Dataset):
    def __init__(self, all_mfcc, all_targets, b_set):
        self.all_mfcc = all_mfcc
        self.all_targets = all_targets
        
        self.wanted_outputs = torch.eye(len(b_set), device=device).double()
        self.output_map = {}
        for i, b in enumerate(b_set):
            self.output_map[b] = self.wanted_outputs[i] # prepare outputs with one 1 at the right place

    def __len__(self):
        return len(self.all_targets)

    def __getitem__(self, idx):
        return self.all_mfcc[idx], self.output_map[self.all_targets[idx]]




class NeuralNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_size)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_n_epochs(train_dataloader, optimizer, model, criterion, n):

    for epoch in range(n):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 20000 == 19999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20000:.3f}')
                running_loss = 0.0




def compute_hmm_nn_b(hmm, nn_model, full_b_set):
    """
    For a sentence hmm model with an attached mfcc, compute b() values
    for every sound frame and every model state, using NN phone model.
    """
    logits = nn_model(hmm.mfcc.double().to(device))
    pred_probab = nn.Softmax(dim=1)(logits)
   
    # Now select b() columns as needed for this hmm
    
    ph_to_i = {ph:i for i, ph in enumerate(full_b_set)} # map phone to column
    
    idx = torch.tensor([ph_to_i[ph] for ph in hmm.b])
    return(pred_probab[:, idx]) # repeat each b() column as needed



def viterbi_align_nn(hmm, nn_model, full_b_set):
    """
    Align hmm states with mfcc, using b_means phone models dictionary
    """
    b = m(compute_hmm_nn_b(hmm, nn_model, full_b_set))
    A = m(hmm.A)
    tmax = hmm.mfcc.size()[0]
    len_x = A.size()[0]
    x_list = [1]+[0]*(len_x-1)
    x_m = m([x_list])
    exponent = 0
    # allocate space for mantissa-like (kept in range) and row-exponent values
    alpha_m = m.rowlist((tmax,len_x))

    for row in range(tmax):
        s = x_m.val.sum() # renormalize
        x_m.val *= 1/s
        exponent += s.log()
        alpha_m[row] = x_m
        x_m = x_m@A*b[row]
    return alpha_m, exponent



