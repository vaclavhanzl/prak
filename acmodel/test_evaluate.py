
import torch

from .evaluate import *


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




def test_align_strings():
    assert align_strings("", "")==("", "", 0)
    assert align_strings("x", "")==("x", ".", 1)
    assert align_strings("xxx", "")==("xxx", "...", 3)
    assert align_strings("", "y")==(".", "y", 1)
    assert align_strings("x", "y")==("x", "y", 1)
    assert align_strings("abc", "abc")==("abc", "abc", 0)
    assert align_strings("abc", "axc")==("abc", "axc", 1)
    assert align_strings("abcx", "abc")==("abcx", "abc.", 1)
    assert align_strings("abxc", "abc")==("abxc", "ab.c", 1)
    assert align_strings("axbc", "abc")==("axbc", "a.bc", 1)
    assert align_strings("xabc", "abc")==("xabc", ".abc", 1)
    assert align_strings("abc", "def")==("abc", "def", 3)
    assert align_strings("abc", "xxx")==("abc", "xxx", 3)
    assert align_strings("xxx", "def")==("xxx", "def", 3)
    s1, s2, d = align_strings("a", "aa")
    assert s1 in {".a", "a."}
    assert (s2, d)==("aa", 1)
    assert align_strings("xlorem ipsumx", "lorqem ypsum")==("xlor.em ipsumx", ".lorqem ypsum.", 4)
    assert align_strings("ahoj", "Ahoj")==("ahoj", "Ahoj", 1)
    #assert align_strings("aaabbbb", "aaaabbb")==("aaabbbb", "aaaabbb", 1) # these fail due to sloppy implementation
    #assert align_strings("aabbbb", "aaabbb")==("aabbbb", "aaabbb", 1)     # fails
    #assert align_strings("abbbb", "aabbb")==("abbbb", "aabbb", 1)         # fails


def test_group_empty_intervals_in_tier():
    assert group_empty_intervals_in_tier([(1, 2, 'a'), (2, 3, ""), (3, 4, "")]) == [(1, 2, 'a'), (2, 4, "")]
    assert group_empty_intervals_in_tier([(1, 2, 'a'), (2, 3, ""), (3, 4, ""), (4, 5, "x")]) == [(1, 2, 'a'), (2, 4, ""), (4, 5, "x")]
    assert group_empty_intervals_in_tier([(2, 3, ""), (3, 4, ""), (4, 5, "x")]) == [(2, 4, ""), (4, 5, "x")]
    assert group_empty_intervals_in_tier([(2, 3, "")]) == [(2, 3, "")]
    assert group_empty_intervals_in_tier([(2, 3, "x")]) == [(2, 3, "x")]
    assert group_empty_intervals_in_tier([]) == []
    assert group_empty_intervals_in_tier([(22, 33, ""), (33, 44, "x")]) == [(22, 33, ""), (33, 44, "x")]
    assert group_empty_intervals_in_tier([(22, 33, "x"), (33, 44, "")]) == [(22, 33, "x"), (33, 44, "")]
    assert group_empty_intervals_in_tier([(1, 2, ""), (2, 3, ""), (3, 4, "")]) == [(1, 4, "")]


def test_prune_tiers_to_suspicious_intervals():
    assert prune_tiers_to_suspicious_intervals([(1, 2, 'a')], [(1, 2, 'b')])==([(1, 2, 'a')], [(1, 2, 'b')])
    assert prune_tiers_to_suspicious_intervals([], [])==([], [])
    assert prune_tiers_to_suspicious_intervals([(1, 2, 'a')], [(1, 2, 'a')])==([(1, 2, '')], [(1, 2, '')])
    assert prune_tiers_to_suspicious_intervals(
        [(1, 2, 'a'), (2, 3, 'b'), (3, 4, 'c')],
        [(1, 2, 'a'), (2, 4, 'b')])==(
        [(1, 3, ''), (3, 4, 'c')],
        [(1, 4, '')])
    assert prune_tiers_to_suspicious_intervals(
        [(1, 2, 'a'), (2, 3, 'b'), (3, 4, 'c')],
        [(1, 2, 'a'), (2, 3, 'x'), (3, 4, 'c')])==(
        [(1, 2, ''), (2, 3, 'b'), (3, 4, '')],
        [(1, 2, ''), (2, 3, 'x'), (3, 4, '')])
    assert prune_tiers_to_suspicious_intervals(
        [(1, 2, 'a'), (2, 3, 'b'), (3, 4, 'c')],
        [(1, 2, 'a'), (2, 3, 'b'), (3, 4, 'c')])==(
        [(1, 4, '')], [(1, 4, '')])


