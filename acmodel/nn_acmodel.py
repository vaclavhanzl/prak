
# NN acoustic models

#from matrix import *
#import numpy as np
#import torch

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import Counter

from prongen.hmm_pron import HMM
# line above fails in pytest, need some other setup
#from .prongen.hmm_pron import HMM


from hmm_acmodel import round_to_two_decimal_digits

from matrix import *

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")


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
    def __init__(self, all_mfcc, all_targets, b_set, sideview = 9):
        self.all_mfcc = all_mfcc
        self.all_targets = all_targets
        self.sideview = sideview
        
        self.wanted_outputs = torch.eye(len(b_set), device=device).double()
        self.output_map = {}
        for i, b in enumerate(b_set):
            self.output_map[b] = self.wanted_outputs[i] # prepare outputs with one 1 at the right place

    def __len__(self):
        return len(self.all_targets) - 2*self.sideview

    def __getitem__(self, idx):
        idx += self.sideview
        return self.all_mfcc[idx-self.sideview:idx+self.sideview+1], self.output_map[self.all_targets[idx]]




class NeuralNetwork(nn.Module):
    def __init__(self, in_size, out_size, mid_size=512):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_size, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, out_size)
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


########### NN INFERENCE ###########

def matrix_extra_edges(A):
    """
    Find states with irregular edge structure (not just enter from left and loop)
    and compute needed correction for token passing.
    """
    #A = torch.tensor(hmm.A)
    extra_edges_from = []
    extra_edges_to = []
    for c in range(1, len(A)): # skip the starting state
        col = A[:,c] # column, how to get to state c
        #NOTE: We might consider WHOLE column to also allow backward edges. REDO THE CODE BELOW?
        assert col[c+1:].sum()==0
        rel = reversed(col[:c+1]) # from diagonal up (we know that below are zeros)
        rel = rel>0 # just yes/no whether token can come to state c from relative states at the left
        if len(rel)>1 and rel[0] and rel[1] and rel[2:].sum()==0: # is it the common case of next/loop?
            continue # nothing to do, common operation will handle this
        rel = torch.tensor(range(len(rel)))[rel] # get offset for every True
        
        # convert to absolute positions:
        rel = c - rel
        extra_edges_from.append(rel)
        extra_edges_to.append(c)
    return extra_edges_from, torch.tensor(extra_edges_to)

#e_e_f, e_e_t = matrix_extra_edges(torch.tensor(hmm.A))
#e_e_f, e_e_t


def next_x(x, extra_edges):
    """
    Pass tokens along edges (incl. loops), taking max() where multiple
    paths arrive at a common state. (Not adding b()s here, this should
    be done with x as a next step.)
    """
    e_e_from, e_e_to = extra_edges
    # Step zero - get irregular edges inputs from the yet unchanged x
    # Save in a vector of values to be put to x later (via scatter)
    if len(e_e_from):
        corrections = torch.stack([x[froms].max() for froms in e_e_from])
        # NOTE: stack hates empty input
    
    
    #loop_boost = 0    
    # Step 1 - align x and one-state-shifted x and take max():
    #s = torch.stack([x[:-1], loop_boost+x[1:]])
    # The stack above can also be made as a special view of x! (not copying any data)
    # MAGIC:   THIS IS TRICKY, REPEATED VALUES ARE "UNDEFINED"
    s = x.as_strided((2, len(x)-1), (1, 1))

    x[1:] = s.max(dim=0).values # pass tokens to the next state and via loops
    # x[0] stays untouched, which corresponds to self-loop

    # Step 2 - put corrections to x
    if len(e_e_from): # if we needed and prepared any corrections
        x.scatter_(0, e_e_to, corrections)  # using IN PLACE version of scatter (with _)
    
#x = torch.zeros(len(hmm.A))
#x[0] = 1
#x[4] = 5
#next_x(x, (e_e_f, e_e_t))

def compute_hmm_nn_log_b(hmm, nn_model, full_b_set, b_log_corr=None):
    """
    For a sentence hmm model with an attached mfcc, compute ln(b()) values
    for every sound frame and every model state, using NN phone model.
    """
    #logits = nn_model(hmm.mfcc.double().to(device)).detach()
    logits = nn_model(hmm.mfcc.double().to(device)).detach().to('cpu')
    pred_probab = nn.LogSoftmax(dim=1)(logits)
    if b_log_corr!=None:
        pred_probab += b_log_corr[None]

    # Now select b() columns as needed for this hmm
    ph_to_i = {ph:i for i, ph in enumerate(full_b_set)} # map phone to column
    
    idx = torch.tensor([ph_to_i[ph] for ph in hmm.b])
    return(pred_probab[:, idx]) # repeat each b() column as needed





def b_log_corrections(tsv_file):
    """
    Compute log(b()) additive correction needed to suppress very frequent
    phones and boost rare ones.
    """
    df = pd.read_csv(tsv_file, sep="\t", keep_default_na=False)
    c=Counter("".join([s for s in df.targets.values]))
    return -torch.tensor([count for phone, count in sorted(i for i in c.items())]).log()





def viterbi_log_align_nn(hmm, nn_model, full_b_set, timrev=False, b_log_corr=None):
    """
    Align hmm states with mfcc, working with logprobs
    """
    b = compute_hmm_nn_log_b(hmm, nn_model, full_b_set, b_log_corr)
    if timrev:
        b = b.flip(0)
    A = hmm.A
    tmax = hmm.mfcc.size()[0]
    len_x = len(A)
    x_list = [0]+[float('-inf')]*(len_x-1)
    x = torch.tensor(x_list)
    alpha = [] #growing list of rows with alpha logprobs
    A = torch.tensor(hmm.A)
    e_e_f, e_e_t = matrix_extra_edges(A) # prepare efficient representation of A
    hmm.optimized_edges = e_e_f, e_e_t  # save it for backward pass - DO THIS ELSEWHERE
    for row in range(tmax):
        s = x.max() #renormalize
        x -= s
        alpha.append(x.clone())
        next_x(x, (e_e_f, e_e_t))
        x += b[row]
    return torch.stack(alpha)


def backward_log_alignment_pass_intervals(hmm, alp):
    """
    Backward pass in normalized logprob alpha.
    Returns alignment intervals to be used for praat.
    The alp matrix is modified in place to show the best path.
    """
    tmax = hmm.mfcc.size()[0]
    len_x = len(hmm.b)
    x_list = [1]+[0]*(len_x-1)
    x = torch.tensor(list(reversed(x_list)))
    frameskip = 0.01 # how many seconds of audio each mfcc row represents
    e_e_f, e_e_t = hmm.optimized_edges
    back = [] # list of backward edges
    back.append(torch.tensor([0])) # just self-loop for the first state
    for i in range(1, len_x):
        back.append(torch.tensor([i, i-1])) # loop and forward edge on the rest of diagonal
    # Now overwrite the irregular ones
    for f, t in zip(e_e_f, e_e_t):
        back[t] = f
    marker = alp.max()+70 # best path marker, just for the image
    last_i = len_x-1        
    end_time = tmax*frameskip
    intervals = []
    for row in range(tmax-1,0-1,-1):
        where_from = back[last_i] # where from we possibly came?
        source_probs = alp[row][where_from] # what logprobs were at source cells?
        ii = source_probs.max(0).indices # find best one among these
        i = back[i][ii] # map it to index in full row
        alp[row, last_i] = marker # indication of the best path in the image

        if i!=last_i: # HMM state change
            phone = hmm.b[last_i]
            begin_time = row*frameskip
            intervals.append((round_to_two_decimal_digits(begin_time), round_to_two_decimal_digits(end_time), phone))
            end_time = begin_time
            last_i = i
            
    phone = hmm.b[0]
    begin_time = 0
    intervals.append((round_to_two_decimal_digits(begin_time), round_to_two_decimal_digits(end_time), phone))
    intervals.reverse()
    return intervals



def triple_sausage_states(sg):
    """
    Triple each phone state in a sausage. This is intended as a primitive duration model
    forcing a minimum duration of phones. Should be complemented by a corresponding
    coalescence of alignment intervals after Viterbi decoding.
    """
    result = []
    for s in sg:
        out_s = set()
        for txt in s:
            txt = "".join(p*3 for p in txt)
            out_s.add(txt)
        result.append(out_s)
    return result
   
#triple_sausage_states([{"abc", 'd'}, {'ee'}])

def triple_hmm_states(hmm):
    """
    Triple each phone state in a HMM. This is intended as a primitive duration model
    forcing a minimum duration of phones. Should be complemented by a corresponding
    coalescence of alignment intervals after Viterbi decoding.
    Expects previously added sausages in HMM. Recomputes A and b.
    """
    hmm.add_sausages(triple_sausage_states(hmm.sausages))

#triple_sausage_states(hmm.sausages)

def group_tripled_intervals(intervals):
    """
    Fix tripling of decoded intervals caused by triple_hmm_states()
    """
    result = []
    while intervals:
        (beg, _, phone), (_, _, p2), (_, end, p3), *intervals = intervals
        assert phone == p2 == p3
        result.append((beg, end, phone))
    return result



def mfcc_add_sideview(mfcc, sideview=9):
    """
    Add few fake frames before and after MFCC so as the window-input NN
    can be computed anywhere.
    At the moment, we fake initial frames by taking frames from the end
    and vice versa (so the MFCC becomes cyclic-like).
    """
    if sideview==0:
        return mfcc
    return torch.cat([mfcc[-sideview:], mfcc, mfcc[0:sideview]], 0)

def mfcc_win_view(mfcc, sideview=9):
    """
    View MFCC as a sequence of NN input windows, sliding window of the
    size 2*sideview+1 over it. The MFCC is expected to be augmented at sides
    using mfcc_add_sideview().
    """
    frames, numceps = mfcc.size()
    winsize = sideview+1+sideview
    sizes = (frames-winsize+1, winsize, numceps)
    strides = numceps, numceps, 1
    return mfcc.as_strided(sizes, strides)



def align_hmm(hmm, model, x_set, b_log_corr, group_tripled=True):
    alp = viterbi_log_align_nn(hmm, model, x_set, b_log_corr=b_log_corr*1.0)
    hmm.intervals = backward_log_alignment_pass_intervals(hmm, alp) # also modifies alp
    hmm.indices = i = alp.max(1).indices
    #s = "".join([hmm.b[ii] for ii in i])
    #hmm.troubling = troubling_alignmet(s) # not working anymore with triple states
    hmm.targets = "".join([hmm.b[ii] for ii in i])
    if group_tripled:
        hmm.intervals = group_tripled_intervals(hmm.intervals)
    return alp # just for debuging, the real result is in hmm.intervals












