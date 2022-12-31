




import sys
sys.path.append("..")
print(sys.path)


import torch

from .nn_acmodel import mfcc_add_sideview, mfcc_win_view


def test_mfcc_add_sideview():
    mfcc = torch.tensor([[1, 2], [3, 4], [5, 6]])
    assert mfcc_add_sideview(mfcc, 0).equal(mfcc)
    assert mfcc_add_sideview(mfcc, 1).equal(torch.tensor([[5, 6], [1, 2], [3, 4], [5, 6], [1, 2]]))
    assert len(mfcc_add_sideview(mfcc, 2))==len(mfcc)+2+2
    assert mfcc_add_sideview(mfcc, 3).equal(torch.cat([mfcc, mfcc, mfcc]))




def test_mfcc_win_view():
    mfcc = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert mfcc_win_view(mfcc, sideview=0)[0].equal(torch.tensor([[1, 2]]))
    assert mfcc_win_view(mfcc, sideview=0)[1].equal(torch.tensor([[3, 4]]))
    assert mfcc_win_view(mfcc, sideview=0)[3].equal(torch.tensor([[7, 8]]))
    assert mfcc_win_view(mfcc, sideview=1)[0].equal(torch.tensor([[1, 2], [3, 4], [5, 6]]))
    assert mfcc_win_view(mfcc, sideview=1)[1].equal(torch.tensor([[3, 4], [5, 6], [7, 8]]))
    








