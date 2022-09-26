
# Copyright © 2022 Václav Hanžl. Part of MIT-licensed https://github.com/vaclavhanzl/prak

# Evaluate alignments


import torch

def dist_matrix_for_strings(vertical_string, horizontal_string):
    """
    Create tensor with 0 where characters match and 1 where not.
    The vertical_string is going top-down along the matrix.
    The horizontal_string is going RIGHT-LEFT along the matrix.
    (Right-left allows use of torch.diagonal() in efficient DWT.)
    """
    vertical = torch.tensor([ord(c) for c in vertical_string]) # top down
    horizontal = torch.tensor([ord(c) for c in reversed(horizontal_string)]) # right left
    return (horizontal[None]!=vertical[:,None]).int()


def dtw_forward_pass(dist, costs_dia_top_right=None):
    """
    Dynamic Time Warping (in fact string index warping) forward pass.
    Given distance matrix, produce cumulated paths costs matrix and
    backtracking directions matrix.
    To enjoy some pytorch speedup, we do it vectorized, by diagonals (!).
    To stay safely in documented behavior of pytorch, we work with views
    made by torch.diagonal() - this dictates rather unusual starting
    point in top-right corner of the matrix, going to bottom-left.
    """
    if costs_dia_top_right!=None:
        costs_dia_top_right = torch.tensor(costs_dia_top_right)[:,None]
        print("costs_dia_top_right feature is unfinished!")

    cum = dist*0-1  # cumulative sums (in fact last 2 diagonals would suffice)
    idx = dist*0-1  # backtracking directions (0=up, 1=up-right, 2=right, -1=start reached)
    rows, cols = dist.size()
    cum[0,-1] = dist[0,-1]
    for i in range(cols-1-1, -rows, -1):
        # both i and ii are biggest at top-right corner, i is relative to top-left corner's diagonal
        ii = i+rows-cols # ii is relative to bottom-right corner's diagonal
        target_cum = cum.diagonal(i) # we will write to cum using this view
        target_idx = idx.diagonal(i) # dtto for idx
        d_1 = cum.diagonal(i+1)    # closest top-right parallel diagonal (distance 1)
        d_2 = cum.diagonal(i+2)    # one more diagonal in distance 2
    
        # Do things slightly differently in upper-right and lower-bottom parts.
        # Starts of vectors are influenced by passing the top-left corner (i),
        # ends of vectors are influenced by passing the bottom-right corner (ii).
        target_cum_full = target_cum
        target_idx_full = target_idx
        if i>=0:
            target_cum = target_cum[1:] # leave out top line (will be done in a special way)
            target_idx = target_idx[1:]
        if ii>=0:
            target_cum = target_cum[:-1] # leave out right line (will be done in a special way)
            target_idx = target_idx[:-1]
        if i<-1:
            d_2 = d_2[1:] # strip first element
        if ii<-1:
            d_2 = d_2[:-1] # strip last element
    
        st = torch.stack([d_2,      # from top-right  0  (most prefered if equal mins)
                          d_1[:-1], # from top        1
                          d_1[1:]   # from right      2
                         ])

        if costs_dia_top_right!=None:
            #print(f"{st.size()=}")
            #print(f"{costs_dia_top_right.size()=}")
            #print(f"{st=}")
            st += costs_dia_top_right
            #print(f"after +, {st=}")

        min_idx = st.min(dim=0) # central point of all this, vectorized min()
        target_cum[:] = min_idx.values
        target_idx[:] = min_idx.indices
    
        if i>=0:
            target_cum_full[0] = d_1[0]    # top line
            target_idx_full[0] = 2         # index
        if ii>=0:
            target_cum_full[-1] = d_1[-1]  # right line
            target_idx_full[-1] = 1        # index
        target_cum_full[:] += dist.diagonal(i) # second of the two vectorized ops used here
    return idx, cum




def dtw_backward_pass(idx):
    """
    Backtrack best alignment path using idx returned by dtw_forward_pass().
    Return vector of directions encountered along the best path from the top right
    corner to the bottom left corner (path is reversed after backtracking).
    Directions are encoded like this: 0=vertical, 1=diagonal, 2=horizontal,
    -1=no furthure move (path ends here).
    """
    path = []
    rows, cols = idx.size()
    r, c = rows-1, 0 # starting in lower left corner
    while (direction := idx[r,c])!=-1:
        path.append(int(direction))
        match direction:
            case 1: # up
                r -= 1
            case 0: # up-right
                r -= 1
                c += 1
            case 2: # right
                c += 1
    assert (r, c) == (0, cols-1)
    path.reverse()
    path.append(-1)
    return path



def fix_ins_del_alignment_in_s1(string_1, string_2, fill_char='.'):
    """
    Scan string_1 for sequences like "..x" and reposition 'x' among dots if
    this achieves correspondance with character in string_2.
    (If no correspondence is found, 'x' is moved in front of dots ("x..") - this
    is a side effect of the simple algorithm used and we do not care.)
    """
    assert len(string_1)==len(string_2)
    string_1 = [*string_1] # convert to list to make it writable
    for i in range(len(string_1)-2, -1, -1): # skip last character, we could not swap
        if string_1[i]==fill_char and string_1[i+1]!=string_2[i+1]: # if not kept by equality
            string_1[i], string_1[i+1] = string_1[i+1], string_1[i] # swap
    return "".join(string_1) # convert back from list to string


def fix_ins_del_alignment(string_1, string_2, fill_char='.'):
    """
    Scan strings for corresponding sequences like "..x" and "abc" (which do
    occur on the output of DTW alignment) and reposition 'x' among dots if
    it corresponds with any character in "abc".
    """
    string_1 = fix_ins_del_alignment_in_s1(string_1, string_2, fill_char=fill_char)
    string_2 = fix_ins_del_alignment_in_s1(string_2, string_1, fill_char=fill_char)
    return string_1, string_2



def align_strings(string_1, string_2, fill_char='.'):
    """
    Insert fill_char into strings 1 and 2 so as they are the same length
    and alignment cost is minimum. Cost includes 1 for every position in
    aligned strings where characters do not match or one of them is fill_char
    (meaning substitution or deletion/insertion).
    """
    dist = dist_matrix_for_strings(string_1, string_2) # vertical, horizontal
    idx, cum = dtw_forward_pass(dist)
    path = dtw_backward_pass(idx)
    #print(f"{path=}")
    #print(f"{dist=}")
    #print(f"{idx=}")
    #print(f"{cum=}")
    aligned_1 = ""
    aligned_2 = ""
    g_1 = (c for c in string_1) # vertical
    g_2 = (c for c in string_2) # horizontal
    for direction in path:
        match direction:
            case 1: # vertical move (insert/delete)
                aligned_1 += next(g_1)
                aligned_2 += fill_char
            case 0|-1: # diagonal move or path end (match or substitute)
                aligned_1 += (c_1 := next(g_1))
                aligned_2 += (c_2 := next(g_2))
            case 2: # horizontal move (insert/delete)
                aligned_1 += fill_char
                aligned_2 += next(g_2)
    aligned_1, aligned_2 = fix_ins_del_alignment(aligned_1, aligned_2, fill_char=fill_char)
    mismatch = 0
    for c1, c2 in zip(aligned_1, aligned_2):
        if c1!=c2:
            mismatch += 1
    return aligned_1, aligned_2, mismatch



def prune_tiers_to_comparable_intervals(tier1, tier2):
    """
    Align phone strings by DTW and keep just intervals which have the same
    phone in both tiers. This alignment is based purely on phone text,
    disregarding times. All phones should have exactly one-character names.
    """
    s1 = "".join([p for _, _, p in tier1])
    s2 = "".join([p for _, _, p in tier2])
    s1, s2, dif = align_strings(s1, s2)
    print(f"{dif=}, {len(s1)=}, {len(tier1)=}")
    assert len(s1)==len(s2)
    g1 = (f_t_p for f_t_p in tier1)
    g2 = (f_t_p for f_t_p in tier2)
    out1 = []
    out2 = []
    for p1, p2 in zip(s1, s2):
        assert p1!='.' or p2!='.' # dot marks deletion in the other string, cannot be in both
        if p1=='.':
            _,_,p = next(g2) # discard in tier2 what corresponds to dot in (aligned) s1
            assert p==p2
            continue
        if p2=='.':
            _,_,p = next(g1) # dtto vice versa
            assert p==p1
            continue
        if p1!=p2:
            _,_,pg1 = next(g1) # discard both if aligned phones differ (substitution)
            _,_,pg2 = next(g2)
            assert pg1==p1
            assert pg2==p2
            continue
        out1.append(next(g1))
        out2.append(next(g2))
    return out1, out2


def compare_tier_times(tier1, tier2):
    """
    Compare times of two tiers with identical phones
    """
    dif = 0
    mid_dif = 0
    for i, ((b1, e1, p1), (b2, e2, p2)) in enumerate(zip(tier1, tier2)):
        #print((p1, p2))
        #assert p1==p2
        if p1!=p2:
            #print(f"{i} phones matched, then '{p1}'!='{p2}'")
            break
        dif += abs(b1-b2)+abs(e1-e2)
        mid_dif += abs((b1+e1)/2-(b2+e2)/2)
    if i==0:
        return 0, 100, 100, 'xx'
    return i, mid_dif/i, dif/i/2, p1+p2



