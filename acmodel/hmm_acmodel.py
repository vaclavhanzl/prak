
# GMM acoustic models - b() functions, Baum-Welch, Viterbi etc.

from matrix import *
import math  # for pi

from gmm_model_data import global_var, b_means


def compute_b(ref_means, ref_vars, means):
    """
    Compute probability b() for every combination of reference mean
    (one reference per phome) and pronunciation frame (0.01s segment
    described by cepstrum).
    The ref_vars argument can be either per-phone (as ref_means) or
    common for all phones (in this case it is broadcasted).
    """
    sum = ( (ref_means[None]-means[:,None])**2/ref_vars ).sum(dim=2)

    pi = torch.tensor(math.pi)
    n = global_var.size()[0]
    #g = (global_var.prod()*(2*pi)**n)**-0.5
    g = 1
    b_probs = g * (-0.5*sum).exp()
    return b_probs



def compute_hmm_b(hmm, means_dict):
    """
    For a sentence hmm model with an attached mfcc, compute b() values
    for every sound frame and every model state.
    """
    b_set = sorted({*hmm.b}) # which phone's b()s will we need
    #print(b_set)
    
    b_set_means = torch.cat([b_means[ph][None] for ph in b_set]) # matrix of all needed means

    #b_means_cat = torch.cat([b_means[ph][None] for ph in hmm.b]) # means repeated as hmm.b dictates
    cb = compute_b(b_set_means, global_var, hmm.mfcc) # compute all needed b()s but each only once
    #print(cb.size())
    # Now repeat each b() column as needed for this hmm
    
    ph_to_i = {ph:i for i, ph in enumerate(b_set)} # map phone to column
    #print(f"{ph_to_i=}")
    
    idx = torch.tensor([ph_to_i[ph] for ph in hmm.b])
    #print(f"{idx=}")
    
    return(cb[:, idx]) # repeat each b() column as needed


base = 1000

def compute_alpha(hmm, b):
    """
    Compute alpha for hmm with mfcc, using b matrix made with corrent phone models
    """
    A = m(hmm.A)
    tmax = hmm.mfcc.size()[0]
    len_x = A.size()[0]
    x_list = [1]+[0]*(len_x-1)
    x_m = m([x_list])
    exponent = 0
    # allocate space for mantissa-like (kept in range) and row-exponent values
    alpha_m = m.rowlist((tmax,len_x))
    alpha_exp = m.rowlist((tmax,1))

    for row in range(tmax):
        while x_m.max()<1/base: # renormalize and remember power of base used
            x_m *= base
            exponent -= 1
        alpha_exp[row] = exponent
        alpha_m[row] = x_m
        x_m = x_m@A*b[row]
    return alpha_m, alpha_exp


def compute_beta(hmm, b):
    """
    Compute beta for hmm with mfcc, using b matrix made with corrent phone models
    """
    At = m(hmm.A).T()
    tmax = hmm.mfcc.size()[0]
    len_x = At.size()[0]
    x_list = [1]+[0]*(len_x-1)
    x_m = m([list(reversed(x_list))])
    exponent = 0
    beta_m = m.rowlist((tmax,len_x))
    beta_exp = m.rowlist((tmax,1))

    for row in range(tmax-1,0-1,-1):
        beta_m[row] = x_m
        beta_exp[row] = exponent
        x_m = x_m@At*b[row]
        while x_m.max()<1/base: # renormalize and remember power of base used
            x_m *= base
            exponent -= 1
    return beta_m, beta_exp


#NOTE: We could in fact get rid of all the exponents in alpha, beta and L cause 
# the final normalization will throw them anyway!
def compute_normalized_L(hmm, b):
    alpha_m, alpha_exp = compute_alpha(hmm, b)
    beta_m, beta_exp = compute_beta(hmm, b)
    L_m = alpha_m*beta_m # this is .*
    L_exp = alpha_exp+beta_exp
    L_exp += float(-L_exp.max())
    # re-normalize L - in fact not needed, we will normalize to sum=1 in rows anyway
    tmax = L_m.size()[0]
    for row in range(tmax):
        while float(L_exp[row].val)<0: # renormalize and remember power of base used
            L_m[row] *= 1/base  # invokes setitem which converts L_m to dense
            L_exp[row] += 1
    X = L_m.val * (1/L_m.val.sum(dim=1))[:,None]
    return X # Normalized L with sum=1 for each row. Only good for alignment (recognition score lost)

#NOTE: compute prob per frame accs in L_v2

def add_to_distribution_statistics(acc_means, acc_weights, hmm, L):
    """
    Fore one training sentence aligned with its hmm using the current phone models,
    accumulate statistics for later computition of the new phone models.
    The 'L' matrix is expected to be normalized to sum 1 in each row.
    Columns of the L matrix correspond to elements of hmm.b (phone states).
    Rows of the L matrix correspond to rows in hmm.mfcc (0.01s time frames).
    """
    means = (L[:,None]*hmm.mfcc[:,:,None]).sum(dim=0) # sizes like [363, 1, 35] and [363, 13, 1]
    #print(means.size())
    # yet to be summed to phonemes
    weights = L.sum(dim=0)
    #print(weights.size())
    for i, phone in enumerate(hmm.b): # now b is string, may be a list of strings later
        if phone not in acc_means:
            acc_means[phone] = 0 # will be broadcast to the right size
            acc_weights[phone] = 0
        acc_means[phone] += means[:,i]
        acc_weights[phone] += weights[i]



def viterbi_align(hmm, b_means):
    """
    Align hmm states with mfcc, using b_means phone models dictionary
    """
    #b = m(compute_hmm_b(hmm, b_means).clamp(min=0.00001))
    b = m(compute_hmm_b(hmm, b_means))
  
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

# NOTE: Try max instead of +



def backward_alignment_pass(hmm, alp):
    """
    Backward pass in normalized alpha, giving a Viterbi/B-W hybrid (as the alpha
    was computed using all paths, not just the best one).
    """
    At = m(hmm.A).T()
    tmax = hmm.mfcc.size()[0]
    len_x = At.size()[0]
    x_list = [1]+[0]*(len_x-1)
    x_m = m([list(reversed(x_list))])

    bap = m.rowlist((tmax,len_x)) # backward alignment pass trace, one 1 among 0s in each row

    for row in range(tmax-1,0-1,-1):
        bap[row] = x_m # like: 0 0 0 1 0 0
        x_m = x_m@At * m(alp[row,:]) # got fuzzy now, need to re-focus on 1 best
        i = x_m.val.max(1).indices[0]
        x_m.val *= 0
        x_m.val[0,i] = 1
    return torch.cat(bap.val)


def round_to_two_decimal_digits(x):
    """
    Slightly change the value of x so that it is likely printed with at most two
    decimal digits, even for values like 35*0.01 (=0.35000000000000003).
    """
    x = "%.2f" % x
    x = float(x)
    return x


def backward_alignment_pass_intervals(hmm, alp):
    """
    Backward pass in normalized alpha, giving a Viterbi/B-W hybrid (as the alpha
    was computed using all paths, not just the best one).
    Return alignment intervals to be used for praat.
    """
    At = m(hmm.A).T()
    tmax = hmm.mfcc.size()[0]
    len_x = At.size()[0]
    x_list = [1]+[0]*(len_x-1)
    x_m = m([list(reversed(x_list))])
    frameskip = 0.01 # how many seconds of audio each mfcc row represents

    bap = m.rowlist((tmax,len_x)) # backward alignment pass trace, one 1 among 0s in each row

    last_i = len(hmm.b)-1
    end_time = tmax*frameskip
    intervals = []
    for row in range(tmax-1,0-1,-1):
        bap[row] = x_m # like: 0 0 0 1 0 0
        x_m = x_m@At * m(alp[row,:]) # got fuzzy now, need to re-focus on 1 best
        i = x_m.val.max(1).indices[0]
        x_m.val *= 0
        x_m.val[0,i] = 1
        
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
    #return torch.cat(bap.val)
    return intervals, torch.cat(bap.val)



def troubling_alignmet(s):
    """
    Count cases of a 1-frame only for state alignment (these are suspicious).
    Overall sum of troubling alignments hints quality of the acoustic model.
    """
    troubles = ""
    for i in range(1,len(s)-1):
        if s[i-1]!=s[i]!=s[i+1]:
            troubles += s[i]
    return troubles






















