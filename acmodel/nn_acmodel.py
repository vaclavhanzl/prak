
# Copyright © 2022 Václav Hanžl. Part of MIT-licensed https://github.com/vaclavhanzl/prak

# NN acoustic models


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import Counter

# We need pandas only for NN AM training. Avoid dependency for just the inference
# using a standard model. FIXME, get rid of the need to comment/uncomment.
#import pandas as pd


#print(f"{__name__=}")
#print(f"{__package__=}")

#from ..prongen.hmm_pron import HMM
from prongen.hmm_pron import HMM




# line above fails in pytest, need some other setup
#from .prongen.hmm_pron import HMM


from .hmm_acmodel import round_to_two_decimal_digits
from .matrix import *
from .praat_ifc import read_word_tier_from_textgrid_file




# COMMENTED OUT BELOW FOR TEXTGRID TESTS
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

#print(f"Using {device} device")

b1_string  = "?AEGHNOZabcdefghjklmnoprstuvyz|áéóúýčďňŘřšťŽž" # phones used by our AC models
b2_string = "ȜĀĒĜĤŅŌŹāƃȼƌēƒĝħĵĸłƜńōƤŕşŦūɅȳź|ǎĕŏŭƿĉȢǹŔȑŝţȤȥ" # names for state 2
b3_string = "ȝĂĖĠĦŃǑŻȃƀċđėſġĥȷķļȸņȯƥŗșŧȗɄƴż|ąęőũŷćƍŉŖȓśțǮǯ" # names for state 3 (note that '|' is repeated in all 3)
b123_string = b1_string+(b2_string+b3_string).replace('|','')

b1_set = [*b1_string] # (not a set in fact)
b2_set = [*b2_string]
b3_set = [*b3_string]
b123_set = [*b123_string] # replacement for b_set in untied tristate models (phone composed of 3 different states, only '|' has 3 tied states)

phone_states = {} # maps phone to string of states (for untied tristate models)
for p, p2, p3 in zip(b1_set, b2_set, b3_set):
    phone_states[p] = p+p2+p3
phone_states['_'] = '___' # hand-fix beeded for untied tristate phones and word separators introduced for word tier


b_set = b1_set # Default 45 phone set, can be changed to 133 state b123_set




def get_training_hmms(nn_train_tsv_file, derivatives=2):
    import pandas as pd # deffered till really needed (just for training)
    df = pd.read_csv(nn_train_tsv_file, sep="\t", keep_default_na=False)
    hmms = []
    if 'targets' in df.keys():
        for wav, sentence, targets in list(zip(df.wav.values, df.sentence.values, df.targets.values)):
            hmm = HMM(sentence, wav=wav, derivatives=derivatives)
            hmm.targets = targets
            hmms.append(hmm)
    else:
        for wav, sentence in list(zip(df.wav.values, df.sentence.values)):
            hmm = HMM(sentence, wav=wav, derivatives=derivatives)
            hmm.targets = None
            hmms.append(hmm)
    return hmms


def collect_training_material(hmms):
    #b_set = sorted({*"".join([hmm.b for hmm in hmms ])}) # make sorted set of all phone names in the training set
    #out_size = len(b_set)
    #in_size = hmms[0].mfcc.size(1)
    all_targets = "".join([hmm.targets for hmm in hmms])
    train_len = len(all_targets)
    all_mfcc = torch.cat([hmm.mfcc for hmm in hmms]).double().to(device)
    assert all_mfcc.size()[0]==train_len
    return all_mfcc, all_targets #, b_set # FIXED: Do not return b_set, we do not compute it here anymore


def mfcc_make_speaker_vector(mfcc):
    """
    Create simple speaker/environment/mic describing vectors for adaptation.
    Make average cepstra over the recording (very simple i-vector like approach).
    Split MFCC to 3 groups according to log-energy (above average is one group,
    below average is further split into two groups the same way) and stack these
    3 vectors to a matrix (which can be concatenated with a MFCC window).
    """
    energy = mfcc[:, 0]
    mean = energy.mean()
    low_mean = energy[energy<mean].mean()
    high_mean = energy[energy>=mean].mean()

    mfcc_lowest = mfcc[energy<low_mean].mean(dim=0)
    mfcc_lower = mfcc[(energy>=low_mean)&(energy<mean)].mean(dim=0)
    mfcc_higher = mfcc[(energy>=mean)&(energy<high_mean)].mean(dim=0)
    mfcc_highest = mfcc[energy>=high_mean].mean(dim=0)

    return torch.stack([mfcc_lowest, mfcc_lower, mfcc_higher, mfcc_highest])
    # NOTE: This acts as ONE s-vector, even if formed 4*13




class SpeechDataset(Dataset):
    def __init__(self, all_mfcc, all_targets, b_set, sideview = 9, speaker_vectors = None):
        self.all_mfcc = all_mfcc
        self.all_targets = all_targets
        self.sideview = sideview
        self.speaker_vectors = speaker_vectors
        
        self.wanted_outputs = torch.eye(len(b_set), device=device).double()
        self.output_map = {}
        for i, b in enumerate(b_set):
            self.output_map[b] = self.wanted_outputs[i] # prepare outputs with one 1 at the right place

    def __len__(self):
        return len(self.all_targets) - 2*self.sideview

    def __getitem__(self, idx):
        idx += self.sideview
        mfcc_window = self.all_mfcc[idx-self.sideview:idx+self.sideview+1]

        nn_input = mfcc_window
        
        if self.speaker_vectors!=None:
            speaker_vector = self.speaker_vectors[idx]
            nn_input = torch.cat([mfcc_window, speaker_vector])
    
        return nn_input, self.output_map[self.all_targets[idx]]



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



def make_nn_input_from_mfcc_and_s_vec(mfcc, s_vec):
    """
    Create NN batch-input from MFCC and speaker vector. The MFCC can be either
    a plain len*13 matrix or a tensor view simulating MFCC windows.
    Currently only the second option is supported.
    """
    # s_vec has torch.Size([4, 13])
    # s_vec[None] has torch.Size([1, 4, 13])
    # torch.cat() does not broadcast, we need explicit torch.expand()
    #
    #                             mfcc can have e.g. torch.Size([432, 19, 13])
    ex = s_vec[None].expand(len(mfcc),-1, -1) # e.g. torch.Size([432,  4, 13])
    return torch.cat([mfcc, ex], dim=1) #       e.g. torch.Size([432, 23, 13])




def compute_hmm_nn_log_b(hmm, nn_model, full_b_set, b_log_corr=None):
    """
    For a sentence hmm model with an attached mfcc, compute ln(b()) values
    for every sound frame and every model state, using NN phone model.
    """
    #logits = nn_model(hmm.mfcc.double().to(device)).detach()
    if hmm.speaker_vector==None:
        nn_input = hmm.mfcc
    else:
        nn_input = make_nn_input_from_mfcc_and_s_vec(hmm.mfcc, hmm.speaker_vector)

    logits = nn_model(nn_input.double().to(device)).detach().to('cpu')
    pred_probab = nn.LogSoftmax(dim=1)(logits)
    if b_log_corr!=None:
        pred_probab += b_log_corr[None]

    # Now select b() columns as needed for this hmm
    ph_to_i = {ph:i for i, ph in enumerate(full_b_set)} # map phone to column
    
    idx = torch.tensor([ph_to_i[ph] for ph in hmm.b])
    return(pred_probab[:, idx]) # repeat each b() column as needed





def b_log_corrections(tsv_file, b_set=b1_set):
    """
    Compute log(b()) additive correction needed to suppress very frequent
    phones and boost rare ones.
    """
    #print(tsv_file)
    if tsv_file.endswith("sv200c-100_training_0024.tsv") or tsv_file.endswith("half.tsv") or tsv_file.endswith("bfloat16.tsv"): # avoid dependency on tsv and pandas
        return torch.tensor([-11.1186,  -7.9824,  -7.1365,  -9.1255, -10.7120,  -8.5094, -10.6577,
                             -7.4565, -12.0110, -11.0539, -11.4070, -11.4735, -12.1724, -10.4384,
                             -9.7028, -10.7073, -11.6203, -11.7481, -11.9372, -11.5911, -11.9380,
                             -12.0138, -11.6192, -11.7821, -12.4654, -12.0538, -11.4064, -11.7344,
                             -11.9744, -11.5930, -14.1668, -11.7700, -11.0398,  -7.3505, -10.6392,
                             -11.8518, -11.1380, -10.3673, -11.5502, -10.7025, -10.2276, -10.9719,
                             -10.6571,  -5.9738, -10.7126])

    import pandas as pd # deffered till really needed (just for training, or testing new models)
    df = pd.read_csv(tsv_file, sep="\t", keep_default_na=False)
    c=Counter("".join([s for s in df.targets.values]))
    return -torch.tensor([c[phone] for phone in b_set]).log()



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



def triple_sausage_states(sg, untied=False):
    """
    Triple each phone state in a sausage. This is intended as a primitive duration model
    forcing a minimum duration of phones. Should be complemented by a corresponding
    coalescence of alignment intervals after Viterbi decoding.
    For untied=True, the 3 states get individual b()s, except '|' which is still tied.
    """
    result = []
    for s in sg:
        out_s = set()
        for txt in s:
            if not untied:
                txt = "".join(p*3 for p in txt)
            else:
                txt = "".join(phone_states[p] for p in txt)
            out_s.add(txt)
        result.append(out_s)
    return result


# Number of states for phones. Groups separated by dots get min. durations 0(_), 1, 2(j), 3, ..., 9(á)
# (The _ is needed for varstate experiments)
phone_min_duration_string = "_..j.?GNdehlnoruvyň.abfgmtúý.Hpz|ďŘřž.Zkséóšť.AŽ.OEcč.á"

phone_num_states = {}
for num_states, phones in enumerate(phone_min_duration_string.split('.')):
    for p in phones:
        phone_num_states[p] = num_states


def multiply_sausage_states(sg, phone_num_states=phone_num_states):
    """
    Multiply each phone state in a sausage according to a minimum acceptable duration, e.g.
    to disallow short lengths in a 5% quantile in hand-labeled data. Should be complemented
    by a corresponding coalescence of alignment intervals after Viterbi decoding.
    """
    result = []
    for s in sg:
        out_s = set()
        for txt in s:
            txt = "".join(p*phone_num_states[p] for p in txt)
            out_s.add(txt)
        result.append(out_s)
    return result
   
#triple_sausage_states([{"abc", 'd'}, {'ee'}])

def triple_hmm_states(hmm, untied=False):
    """
    Triple each phone state in a HMM. This is intended as a primitive duration model
    forcing a minimum duration of phones. Should be complemented by a corresponding
    coalescence of alignment intervals after Viterbi decoding.
    Expects previously added sausages in HMM. Recomputes A and b.
    For untied=True, non-silent phones get 3 different untied states.
    """
    hmm.add_sausages(triple_sausage_states(hmm.sausages, untied))

def multiply_hmm_states(hmm, phone_num_states=phone_num_states):
    """
    Multiply each phone state in a HMM according to a minimum acceptable duration, e.g.
    to disallow short lengths in a 5% quantile in hand-labeled data. This is intended as
    a primitive duration model forcing a minimum duration of phones. Should be complemented
    by a corresponding coalescence of alignment intervals after Viterbi decoding.
    Expects previously added sausages in HMM. Recomputes A and b.
    """
    hmm.add_sausages(multiply_sausage_states(hmm.sausages, phone_num_states))

#triple_sausage_states(hmm.sausages)

def group_tripled_intervals(intervals):
    """
    Fix tripling of decoded intervals caused by triple_hmm_states()
    """
    result = []
    while intervals:
        (beg, _, phone), (_, _, p2), (_, end, p3), *intervals = intervals
        #assert phone == p2 == p3    # THIS ASSERT SHOULD BE AVOIDED FOR UNTIED TRISTATE CASE
        result.append((beg, end, phone))
    return result

def group_multiplied_intervals(intervals, phone_num_states=phone_num_states):
    """
    Fix multiplication of decoded intervals caused by multiply_hmm_states()
    """
    result = []
    while intervals:
        (beg, end, phone), *intervals = intervals
        for _ in range(1, phone_num_states[phone]):
            (b2, e2, p2), *intervals = intervals
            assert p2==phone
            assert b2==end
            end = e2
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
    hmm.targets = "".join([hmm.b[ii] for ii in i])
    if group_tripled:
        hmm.intervals = group_tripled_intervals(hmm.intervals)
    return alp # just for debuging, the real result is in hmm.intervals





class Dotaccess():
    """
    Primitive class allowing .x access instead of ["x"].
    Should be created from a dictionary.
    """
    def __init__(self, d):
        self.__dict__ = d
    def __repr__(self):
        return str(self.__dict__)


inference_device = "cpu"


def load_nn_acoustic_model(filename_base, mid_size=100, varstates=True, morestates=True, adapt=True, corr_weight=1.0, special_corr=None, b_set=b1_set):
    """
    Prepare everything needed for decoding with a given NN AM.
    Using most common defaults from latest good trainings. (These
    should rather be stored with the NN model.)
    'morestates' means triple or variable states
    'corr_weight' is weight of correction based on phone frame statistic
    'special_corr' is a dict with some alternative correction
    Specify b_set=b123_set for untied tristate models.
    """
    sideview = 9
    if adapt:
        s_vector_size = 4
    else:
        s_vector_size = 0
    mfcc_size = 13
    in_size = (sideview+1+sideview + s_vector_size)*mfcc_size
    out_size = len(b_set)
    if special_corr is None:
        b_log_corr = b_log_corrections(filename_base+".tsv", b_set=b_set)
    else:
        b_log_corr = special_corr
    loc = locals() # Snapshot arguments and additional interesting parameters
    
    model = NeuralNetwork(in_size, out_size, mid_size).to(inference_device)
    model.par = Dotaccess(loc) # Later we can get e.g.: model.par.mid_size
    
    model.load_state_dict(torch.load(filename_base+".pth", map_location=torch.device('cpu')))
    model.eval()
    return model



show_hmm = None # temporary debug tool, to be removed later

def align_wav_file_using_model(wav_file, model):
    """
    Read wav file and corresponding textgrid, get text from word tier,
    compute our own phone alignment using given NN model, evaluate.
    """
    suffix = ".wav"
    assert wav_file.endswith(suffix)
    filename_base = wav_file[:-len(suffix)]
    word_tier = read_word_tier_from_textgrid_file(filename_base+".TextGrid")
    txt = " ".join(word for _, _, word in word_tier)
    hmm = HMM(txt, wav_file, derivatives=0)
    if model.par.varstates:
        multiply_hmm_states(hmm)
    elif model.par.morestates:
        triple_hmm_states(hmm)
    if model.par.adapt:
        hmm.speaker_vector = mfcc_make_speaker_vector(hmm.mfcc)
    else:
        hmm.speaker_vector = None
    hmm.mfcc = mfcc_win_view(mfcc_add_sideview(hmm.mfcc, sideview=model.par.sideview), sideview=model.par.sideview)

    alp = align_hmm(hmm, model, b_set, b_log_corr=model.par.b_log_corr*model.par.corr_weight, group_tripled=model.par.morestates and not model.par.varstates)
    if model.par.varstates:
        hmm.intervals = group_multiplied_intervals(hmm.intervals)

    global show_hmm; show_hmm = hmm

    return hmm.intervals


def generator_is_empty(gen):
    """
    Return True if nothing is left in the generator.
    Intended for use in assert. Test is destructive,
    cannot be used when any remaining value is needed.
    """
    try:
        next(gen)
        return False
    except StopIteration:
        return True

def compute_word_tier(hmm):
    """
    For aligned HMM with phone tier, compute also the word tier.
    """
    # This procedure is awfuly tricky. We simplified things elsewhere by
    # not using non-emitting states and not using any wFST infrastructure.
    # Overall the balance is still positive but here we pay the debt.
    # For each state in hmm.b, we know to which word in sequence it belongs.
    # This is not exactly an easy-to-work-with form of word information
    # but we clench our teeth and do it here!
    
    
    #last_idx = 0 # FIXME: when state-alignment slight occasional discrepancy if fixed, we may need a change here
    last_p_i = None
    last_w_i = None
    wi_begin = None
    wi_word = None
    
    phone_intervals = (p_i for p_i in hmm.intervals)
    word_intervals = []
    
    # NOTE: Times should be approx. i/100 but we rather derive them from times already computed for phones.
    #       This way e.g. cepstrum-based focusing will propagate to word times as well.
    for i, idx in enumerate(hmm.indices):
        idx = int(idx) # mainly to avoid torch warning about floordiv
        phone = hmm.b[idx]
        w_i = hmm.word_indices[idx]//3  # FIXME: ONLY WORKS FOR TRIPLED STATES
        
        if phone=='|':
            word = ""
            w_i = -1 # '|' has w_i of the following word but we want it to be treated as a separate word instead
        else:
            if w_i<len(hmm.words):
                word = hmm.words[w_i]
            else:
                word = "ERROR" # Our word list got out of sync with the pron list!!! FIXME

        p_i = idx//3  # FIXME: ONLY WORKS FOR TRIPLED STATES
            
        if p_i!=last_p_i: # HMM state change
            pi_begin, pi_end, pi_phone = next(phone_intervals) # consume phone intervals in sync with state changes
            #print((pi_begin, pi_end, pi_phone))
            last_p_i = p_i    
        
            if w_i!=last_w_i or pi_phone=='|': # just coming phone interval already belongs to a new word or silence interval
                if wi_begin!=None: # we have preceding word interval to finish
                    wi_end = pi_begin
                    word_intervals.append((wi_begin, wi_end, wi_word)) # complete another word tier interval
                    #print(("--------------", wi_begin, wi_end, wi_word))
        
                wi_begin = pi_begin
                last_w_i = w_i

        #print(i, idx, p_i, w_i, phone, word)
        wi_word = word # remember last word in case we need to emit it the next time

    
    word_intervals.append((wi_begin, pi_end, wi_word)) # complete last silence in the word tier interval
    #print(("--------------", wi_begin, pi_end, wi_word))
    assert generator_is_empty(phone_intervals)
    hmm.word_intervals = word_intervals

#compute_word_tier(show_hmm)

def align_wav_and_text_using_model(wav_file, txt, model):
    """
    Read wav file and align it with text.
    """
    suffix = ".wav"
    assert wav_file.endswith(suffix)
    filename_base = wav_file[:-len(suffix)]
    #word_tier = read_word_tier_from_textgrid_file(filename_base+".TextGrid")
    #txt = " ".join(word for _, _, word in word_tier)
    hmm = HMM(txt, wav_file, derivatives=0)
    global show_hmm; show_hmm = hmm

    hmm.b_set = b_set
    if model.par.out_size>45:  # FIXME, HACK
        triple_hmm_states(hmm, untied=True)
        hmm.b_set = b123_set # FINISHME
    elif model.par.varstates:
        multiply_hmm_states(hmm)
    elif model.par.morestates:
        triple_hmm_states(hmm)
    if model.par.adapt:
        hmm.speaker_vector = mfcc_make_speaker_vector(hmm.mfcc)
    else:
        hmm.speaker_vector = None
    hmm.mfcc = mfcc_win_view(mfcc_add_sideview(hmm.mfcc, sideview=model.par.sideview), sideview=model.par.sideview)

    alp = align_hmm(hmm, model, hmm.b_set, b_log_corr=model.par.b_log_corr*model.par.corr_weight, group_tripled=model.par.morestates and not model.par.varstates)
    if model.par.varstates:
        hmm.intervals = group_multiplied_intervals(hmm.intervals)

        # Hack for phone measurements with varstate models:
        return hmm.intervals, hmm.intervals # DO NOT COMPUTE WORD INTERVALS, BROKEN

    compute_word_tier(hmm)
        
    return hmm.intervals, hmm.word_intervals

















