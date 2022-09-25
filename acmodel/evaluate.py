
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


def dtw_forward_pass(dist):
    """
    Dynamic Time Warping (in fact string index warping) forward pass.
    Given distance matrix, produce cumulated paths costs matrix and
    backtracking directions matrix.
    To enjoy some pytorch speedup, we do it vectorized, by diagonals (!).
    To stay safely in documented behavior of pytorch, we work with views
    made by torch.diagonal() - this dictates rather unusual starting
    point in top-right corner of the matrix, going to bottom-left.
    """
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
