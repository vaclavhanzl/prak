
import torch

from .evaluate import dtw_forward_pass, fix_ins_del_alignment_in_s1


def test0_dtw_forward_pass():
    dist = torch.tensor([[1]])
    idx, cum = dtw_forward_pass(dist)
    assert idx.equal(torch.tensor([[-1]]))
    assert cum.equal(torch.tensor([[1]]))

def test1_dtw_forward_pass():
    dist = torch.tensor([[1, 2], [3, 4]])
    idx, cum = dtw_forward_pass(dist)
    assert idx.equal(torch.tensor([[2, -1], [0, 1]]))
    assert cum.equal(torch.tensor([[3, 2], [5, 6]]))

def test2_dtw_forward_pass():
    dist = torch.tensor([[1, 0], [0, 4]])
    idx, cum = dtw_forward_pass(dist)
    assert idx.equal(torch.tensor([[2, -1], [0, 1]]))
    assert cum.equal(torch.tensor([[1, 0], [0, 4]]))

def test3_dtw_forward_pass():
    dist = torch.tensor([[1, 0], [0, 4], [0, 7]])
    idx, cum = dtw_forward_pass(dist)
    assert idx.equal(torch.tensor([[2, -1], [0, 1], [1, 1]]))
    assert cum.equal(torch.tensor([[1, 0], [0, 4], [0, 11]]))

def test4_dtw_forward_pass():
    dist = torch.tensor([[7, 8, 1],
                         [1, 2, 3]])
    idx, cum = dtw_forward_pass(dist)
    assert idx.equal(torch.tensor([[ 2,  2, -1],
                                   [ 2,  0,  1]]))
    assert cum.equal(torch.tensor([[16,  9,  1],
                                   [ 4,  3,  4]]))


def test5_dtw_forward_pass():
    dist = torch.arange(5*8).view([5,8])*31%7
    #[[0, 3, 6, 2, 5, 1, 4, 0],
    # [3, 6, 2, 5, 1, 4, 0, 3],
    # [6, 2, 5, 1, 4, 0, 3, 6],
    # [2, 5, 1, 4, 0, 3, 6, 2],
    # [5, 1, 4, 0, 3, 6, 2, 5]]
    idx, cum = dtw_forward_pass(dist)
    assert cum[4,0]==10

def test6_dtw_forward_pass():
    dist = torch.arange(9*4).view([9,4])*31%7
    #[[0, 3, 6, 2],    11 11  8  2 
    # [5, 1, 4, 0],    12  7  6  2
    # [3, 6, 2, 5],    10 10  4  7
    # [1, 4, 0, 3],     9  8  4 10*
    # [6, 2, 5, 1],    12  6  9 11
    # [4, 0, 3, 6],    10  6 12 17
    # [2, 5, 1, 4],     8 11 13 21
    # [0, 3, 6, 2],     8 14 19 23
    # [5, 1, 4, 0]]    13 15 23 23*
    idx, cum = dtw_forward_pass(dist)
    assert cum[3].equal(torch.tensor([9, 8, 4, 10]))
    assert cum[-1].equal(torch.tensor([13, 15, 23, 23]))




def test1_fix_ins_del_alignment_in_s1():
    assert fix_ins_del_alignment_in_s1("", "")==""
    assert fix_ins_del_alignment_in_s1("..a", "abc")=="a.."
    assert fix_ins_del_alignment_in_s1("x..a", "xabc")=="xa.."
    assert fix_ins_del_alignment_in_s1("x..abc", "xabcbc")=="xa..bc"
    assert fix_ins_del_alignment_in_s1(".a", "ab")=="a."
    assert fix_ins_del_alignment_in_s1("hello.amello", "helloabmello")=="helloa.mello"
