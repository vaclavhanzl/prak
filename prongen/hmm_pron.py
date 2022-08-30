#! /usr/bin/env python3
lib_intro = """
hmm_pron library - generate possible Czech pronunciations as HMM

Copyright (c) 2022 Vaclav Hanzl. This is a free software (see the MIT license).

This file is part of the https://github.com/vaclavhanzl/prak project
"""


import sys
import os
import torchaudio # for mfcc
import torch

if (__name__ == '__main__'): # messing with path to make imports work when this is a script
    sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from prongen.prak_prongen import *
from acmodel import matrix




# line_iterable_to_lexirules() cannot make this because of space being replaced
explicit_spaces = {" ": ["", "|"], "_": [""], "=": [""]}


class HMM:
    """
    Container for a one sentence HMM description.
    First we put just a pronunciation there, later on we may add more
    like cepstrum. It is in fact not only HMM but also the related
    utterance and its acoustic representation.
    """

    def __str__(self):
        """
        Human readable form, used by print()
        """
        s = "------ sentence and HMM ------\n"
        s += f" WAV: {self.wav}\n"
        s += f"ORTO: {self.orto}\n"
        s += self.pretty_pron
        s += "\n"
        s += "   "+" ".join(self.b)
        for p, row in zip(self.b, self.A):
            s += "\n " + p + " " + " ".join([[".", "1"][p] for p in row])
        return s


    def add_sausages(self, sg):
        """
        Create transition matrix A and string of phone labels b.
        Elements of b correspond to columns of A.
        A is indexed [from, to]
        """
        if len(sg[0])!=1:
            print(f"Warning: The very first element has variants (but we can cope with this) {sg=}")
        if len(sg[-1])!=1:
            print(f"Warning/Error: The last element has variants, just one of them will work {sg=}")
        # NOTE: We decided to make a system WITHOUT non-emitting states to keep things simpler.
        #       In alignment, the first state will be a one variant silence so we should be OK.
        #       We do not need to fix things by multiple entries in the initial states vector x
        #       as long as transition probabilities are all the same. But we cannot trick a similar
        #       problem at the end. (We could add no-loop special state with take-anything b...)
        #       Just call this with "|" as the only possibility at begin/end and things will work.
        b = ""
        for s in sg:
            for txt in sorted(s):
                for p in txt:
                    b += p

        A = [[0 for c in b] for r in b] # both rows and columns are as many as phones in b

        row = 0
        dim = len(b)
        ends = [0] # [] would mostly do but fake this in case sg starts by multivariant
        for s in sg:
            new_ends = [] # collect all variant ends here for later connection
            for txt in sorted(s):
                first_in_txt = True
                for e in ends:
                    A[e][row] = 1 # connect to all prev. ends
                for p in txt:
                    A[row][row] = 1 # self loop
                    if not first_in_txt:
                        A[row-1][row] = 1 # connect phones in txt
                    first_in_txt = False
                    row += 1
                new_ends.append(row-1)
            ends = new_ends # in the next variant list, each begin will be connected to these
        self.A = A
        self.b = b
        return self # for any chaining with other ops (like add cepstrum etc.)

    def change_sil_loops(self, sil="|", p=0):
        """
        Disable self-loops in all sil states except the first and last state.
        This is intended to help B-W take off when internal sil states are
        mostly displaced and only slow down learning the sound of silence
        based on the edge regions.
        Loops can be re-instantiated by using p=1.
        """
        _, *inner, _ = enumerate(self.b)
        for i, phone in inner:
            if phone==sil:
                self.A[i][i] = p

    def compute_mfcc(self):
        """
        Load wav file to temporary storage. Compute MFCC and attach it to this object.
        """
        waveform, fs = torchaudio.load(self.wav)
        mfcc = torchaudio.compliance.kaldi.mfcc(waveform, sample_frequency=fs) # default 16kHz is correct for this file

        if True:
            self.mfcc = mfcc
        else:
            mfcc_d = torchaudio.functional.compute_deltas(mfcc)
            mfcc_a = torchaudio.functional.compute_deltas(mfcc_d)
            self.mfcc = torch.cat([mfcc, mfcc_d, mfcc_a], dim=1)



    def __init__(self, sentence=None, wav=None):
        """
        Create HMM model for a training sentence.
        """
        #print(f"{sentence=}")
        args.ctu_phone_symbols = True # TODO: use other mechanism than global args
        args.all_begins = False
        args.all_ends = False

        sen = sentence.strip()
        sg = process(sen, all_begins=args.all_begins, all_ends=args.all_ends)
        sg = [{"|"}] + sg + [{"|"}]
        sg = factor_out_common_begins(sg)
        sg = factor_out_common_ends(sg)
        sg = sausages_replacement_rules(explicit_spaces, sg)
        self.add_sausages(sg)
        self.orto = sen
        self.pretty_pron = prettyprint_sausage([{"PRON: "}] + sg)
        self.wav = wav
        if wav:
            self.compute_mfcc()
            #print(f"Computed mfcc, {self.mfcc.size()=}")


if (__name__ == '__main__' and len(sys.argv)>1 and sys.argv[1]=="--in-jupyter"):
    print("hmm_pron.py library - generate Czech pron HMM. Included to this notebook.")
elif __name__ == '__main__':
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

    """
    sen = sen + " " + sen
    sen = sen + " " + sen
    sen = sen + " " + sen
    sen = sen + " " + sen
    sen = sen + " " + sen
    sen = sen + " " + sen
    sen = sen + " " + sen
    """


    #sen = "kč"

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


    #sg = [{'aa','bb'}, {'ccc','dd','eee'}]


    print(f"{sg=}")


    hmm = HMM().add_sausages(sg)


    hmm.orto = sen
    hmm.pretty_pron = prettyprint_sausage([{"PRON: "}] + sg)
    hmm.wav = "/home/hanzl/f-w/prak/common_voice_cs_23962589.wav"

    print("")
    print(hmm)
