#! /usr/bin/env python3
lib_intro = """
hmm_pron library - generate possible Czech pronunciations as HMM

Copyright (c) 2022 Vaclav Hanzl. This is a free software (see the MIT license).

This file is part of the https://github.com/vaclavhanzl/prak project
"""

from prak_prongen import *


# line_iterable_to_lexirules() cannot make this because of space being replaced
explicit_spaces = {" ": ["", "|"], "_": [""], "=": [""]}

def print_hmm(A, b):
    """
    Print an informative HMM structure
    """
    print("   "+" ".join(b))
    for p, row in zip(b, A):
        print(" " + p + " " + " ".join([p for p in row]))


class HMM:
    """
    Container for a one sentence HMM description.
    First we put just a pronunciation there, later on we may add more.
    """
    pass

def sausages_to_hmm(sg):
    """
    Create transition matrix A and string of phone labels b.
    Elements of b correspond to columns of A.
    A is indexed [from, to]
    """
    if len(sg[0])!=1:
        print(f"Warning: Just one variant of the first element will be accessible. {sg=}")
    #NOTE: We decided to make a system WITHOUT non-emitting states to keep things simpler.
    #      In alignment, the first state will be a one variant silence so we should be OK.
    #      If really needed, theoretically we could fix things by multiple entries in the
    #      initial states vector x (but we'd have to make x part of the HMM description).
    #      To make this theoretical adition easier, we composed A and b into a hmm class
    #      where x could be added later.
    b = ""
    for s in sg:
        for txt in sorted(s):
            for p in txt:
                b += p

    A = [["." for c in b] for r in b] # both rows and columns are as many as phones in b

    row = 0
    dim = len(b)
    ends = [] # no previous ends to connect to
    for s in sg:
        new_ends = [] # collect all variant ends here for later connection
        for txt in sorted(s):
            first_in_txt = True
            for e in ends:
                A[e][row] = "1" # connect to all prev. emds
            for p in txt:
                A[row][row] = "1" # self loop
                if not first_in_txt:
                    A[row-1][row] = "1" # connect phones in txt
                first_in_txt = False
                row += 1
            new_ends.append(row-1)
        ends = new_ends # in the next variant list, each begin will be connected to these
    hmm = HMM()
    hmm.A = A
    hmm.b = b
    return hmm

if (__name__ == '__main__'):
    print("Test of the hmm_pron library - generate Czech pron HMM")

    # simulate some prak_prongen options - we also imported it's args object and can modify it here
    args.ctu_phone_symbols = True
    args.all_begins = False
    args.all_ends = False
    print(f"{args.__dict__=}")

    sen = "jsou oba dva"
    #sen = "panický atak"
    sen = "jsou"
    sen = "a a a"
    sen = "k dohodě došlo již dlouho předtím"
    sen = "kč"

    print(f"{sen=}")

    sen = sen.strip()
    sg = process(sen, all_begins=args.all_begins, all_ends=args.all_ends)
    print(prettyprint_sausage(sg))
    print(f"{sg=}")


    sg = [{"|"}] + sg + [{"|"}]

    print("")
    print(prettyprint_sausage(sg))
    print(f"{sg=}")

    sg = factor_out_common_begins(sg)
    sg = factor_out_common_ends(sg)
    #sg = bme_factorize_sausages(sg)


    sg = sausages_replacement_rules(explicit_spaces, sg)
    print("")
    print(prettyprint_sausage(sg))


    sg = [{'aa','bb'}, {'ccc','dd','eee'}]


    print(f"{sg=}")


    hmm = sausages_to_hmm(sg)

    A = hmm.A
    b = hmm.b

    #print(f"{b=}")

    print("")

    print_hmm(A, b)
