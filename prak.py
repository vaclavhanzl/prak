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


    args = parser.parse_args()

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

    if problems:
        print(f'Prak detected problems. Giving up.', file=sys.stderr)
        sys.exit(1)

    print('Prak, the phonetic alignment tool.')
    print(f'        Input wav: "{args.in_wav}"')
    print(f'   Input TextGrid: "{args.in_tg}" (will look for a phrase tier here)')
    print(f'  Output TextGrid: "{args.out_tg}" (will make phone, word and phrase tiers here)\n')





    model = acmodel.nn_acmodel.load_nn_acoustic_model(f"{base}/acmodel/sv200c-100_training_0024", mid_size=100, varstates=False)

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

    tg_txt = acmodel.praat_ifc.textgrid_file_text(dict(phone=sampa_phone_tier, word=word_tier, phrase=phrase_tier))
    #print(tg_txt)


    with open(args.out_tg, 'w') as f:
        f.write(tg_txt)

    print(f'Alignment written to phone, word and phrase tiers in "{args.out_tg}"')







