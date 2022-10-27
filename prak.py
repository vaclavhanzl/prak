#! /usr/bin/env python3
intro = """
prak - Czech phonetic tool. Makes time aligned pronunciations from text and wav.
Copyright (c) 2022 Vaclav Hanzl. This is a free software (see the MIT license).
This file is part of the https://github.com/vaclavhanzl/prak project.
To learn more, try:

   prak --help
"""

import sys
import os
import argparse

#print(sys.path)

import prongen.hmm_pron
import acmodel.praat_ifc

device = "cpu"
import acmodel.nn_acmodel
# acmodel.evaluate

base = sys.argv[0][:-len('/prak.py')] # base directory of prak
#print(base)

if (__name__ == '__main__'):
    if len(sys.argv)==1: # no argumnents at all
        print(intro)
        sys.exit(0)

    parser = argparse.ArgumentParser(description='Make time aligned pronunciations from Czech text and wav.')


    parser.add_argument('-w','--in-wav', help='Input wav with voice recording', required=True) 

    parser.add_argument('-i','--in-tg', help='Input TextGrid with phrase tier', required=True) 

    parser.add_argument('-o','--out-tg', help='Output TextGrid with phone, word and phrase tiers', required=True) 

    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite output file if it already exists, even if it was also an input file')

    parser.add_argument('-e','--exceptions', help='Text file with additional pronunciation rules') 

    args = parser.parse_args()

    #print(f"{args.exceptions=}")
    #print(f"{prongen.hmm_pron.lexicon_replacements=}")

    problems = False

    if not os.path.exists(args.in_wav):
        print(f'Input wav file "{args.in_wav}" does not exist.', file=sys.stderr)
        problems = True

    if not os.path.exists(args.in_tg):
        print(f'Input TextGrid file "{args.in_tg}" does not exist.', file=sys.stderr)
        problems = True

    if os.path.exists(args.out_tg) and not args.force:
        print(f'Output TextGrid file "{args.out_tg}" already exists.', file=sys.stderr)
        if args.in_tg==args.out_tg:
            print(f'(The same name used for the input and the output file.)', file=sys.stderr)
        print(f'Specify the "-f" or "--force" option to overwrite it.', file=sys.stderr)
        problems = True

    if args.exceptions!=None and not os.path.exists(args.exceptions):
        print(f'Pronunciation exceptions file "{args.exceptions}" does not exist.', file=sys.stderr)
        problems = True

    if problems:
        print(f'Prak detected problems. Giving up.', file=sys.stderr)
        sys.exit(1)

    print('Prak, the phonetic alignment tool.')
    print(f'        Input wav: "{args.in_wav}"')
    print(f'   Input TextGrid: "{args.in_tg}" (will look for a phrase tier here)')
    print(f'  Output TextGrid: "{args.out_tg}" (will make phone, word and phrase tiers here)')

    if args.exceptions!=None:
        print(f'  Exceptions file: "{args.exceptions} (will get there additional pronunciation rules)"')
        additional_rules = prongen.hmm_pron.read_lexirules_table(args.exceptions)
        num_rules_before = len(prongen.hmm_pron.lexicon_replacements)
        prongen.hmm_pron.lexicon_replacements |= additional_rules
        num_rules_after = len(prongen.hmm_pron.lexicon_replacements)
        # print change, so as the user can see number of her own rules, not just examples from exceptions.txt:
        print(f'     (got {len(additional_rules)} rules there, maybe replacing similar or same already known, increasing the total rules number by {num_rules_after-num_rules_before})')
    print("", flush=True) # 'flush' is just an attempt which may not really flush text out now

    model = acmodel.nn_acmodel.load_nn_acoustic_model(f"{base}/acmodel/half", mid_size=100, varstates=False)

    # b prob corr !!!


    in_tiers = acmodel.praat_ifc.read_interval_tiers_from_textgrid_file(args.in_tg)

    if 'phrase' not in in_tiers:
        print(f'Input TextGrid file "{args.in_tg}" does not contain a "phrase" interval tier.', file=sys.stderr)
        print(f'Only these interval tiers found in it: {" ".join(in_tiers.keys())}', file=sys.stderr)
        sys.exit(1)

    phrase_tier = in_tiers['phrase']

    text = " ".join(phrase for (begin, end, phrase) in phrase_tier)

    #print(text)

    phone_tier, word_tier = acmodel.nn_acmodel.align_wav_and_text_using_model(args.in_wav, text, model)

    #print(phone_tier)
    sampa_phone_tier = acmodel.praat_ifc.sampify_tier(phone_tier)

    tg_dict = dict(phone=sampa_phone_tier, word=word_tier, phrase=phrase_tier)

    acmodel.praat_ifc.unify_tier_ends(tg_dict, leading_tier="phrase", max_fuzz=0.1)

    tg_txt = acmodel.praat_ifc.textgrid_file_text(tg_dict)
    #print(tg_txt)

    with open(args.out_tg, 'w', encoding='utf-8') as f: # explicit utf-8 needed on Windows
        f.write(tg_txt)

    print(f'Alignment written to phone, word and phrase tiers in "{args.out_tg}"')







