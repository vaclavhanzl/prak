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


    parser.add_argument('-w','--in-wav', help='Input wav file with voice recording', required=True)

    parser.add_argument('-i','--in-tg', help='Input TextGrid file with phrase tier', required=True)

    parser.add_argument('-o','--out-tg', help='Output TextGrid file with phone, word and phrase tiers', required=True)

    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite output file if it already exists, even if it was also an input file')

    parser.add_argument('-e','--exceptions', help='Text file with additional pronunciation rules') 

    parser.add_argument('-t','--text_tier', default=[], action='extend', nargs='*',
                        help='Name of input tier with text (may repeat for multiple options, first match will be used)')

    parser.add_argument('-k','--keep_tier', default=[], action='extend', nargs='*', help='Keep input tier "old", '
                        'possibly renamed "old:new" or keep all adding suffix by ":suf". Use ":" to just keep all. '
                        'Option may repeat, then for every input tier, all specs will be tried in order till the first match (if any). '
                        'Any colon in tier name can be escaped by backslash.')

    parser.add_argument('-a','--align_tier', default=[], action='extend', nargs='*',
                        help="Rename output tier(s) with alignment from default ['phone', 'word', 'phrase']. "
                        "Same specification as for -k, e.g. '-a phone:ali_phone -a :' or '-a :_auto'. "
                        "Can also prune, e.g. '-a phone' discards 'word' and 'phrase' tiers.")

    args = parser.parse_args()
    if len(args.text_tier)==0:
        args.text_tier = ['phrase', 'Phrase'] # handling default here, otherwise it would be kept even when -t used

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

    if os.path.exists(args.out_tg) and not args.in_tg==args.out_tg:
        print(f'  Removing old pre-existing output TextGrid file "{args.out_tg}".')
        os.remove(args.out_tg)
        # NOTE: This increases our chance that praat script on Windows does notice we crashed later
        # We should rather compare  normalized pathnames to avoid deleting our input when abs/rel names are used.

    if args.exceptions!=None:
        print(f'  Exceptions file: "{args.exceptions}" (additional pronunciation rules)')
        additional_rules = prongen.hmm_pron.read_lexirules_table(args.exceptions)
        num_rules_before = len(prongen.hmm_pron.lexicon_replacements)
        prongen.hmm_pron.lexicon_replacements |= additional_rules
        num_rules_after = len(prongen.hmm_pron.lexicon_replacements)
        # print change, so as the user can see number of her own rules, not just examples from exceptions.txt:
        print(f'     (got {len(additional_rules)} rules there, maybe already known, increasing the total rules number by {num_rules_after-num_rules_before})')
    print("", flush=True) # 'flush' is just an attempt which may not really flush text out now

    model = acmodel.nn_acmodel.load_nn_acoustic_model(f"{base}/acmodel/half", mid_size=100, varstates=False)

    # b prob corr !!!


    in_tiers = acmodel.praat_ifc.read_interval_tiers_from_textgrid_file(args.in_tg)

    for phrase_tier_name in args.text_tier:
        if phrase_tier_name in in_tiers: # find first tier named in -t options
            break
    else: # if no break
        print(f'Input TextGrid file "{args.in_tg}" has no phrase tier named: {", ".join(args.text_tier)}', file=sys.stderr)
        print(f'Only these interval tiers found in it: {", ".join(in_tiers.keys())}', file=sys.stderr)
        sys.exit(1)

    print(f"Tiers {[*in_tiers.keys()]} at input, reading text from '{phrase_tier_name}'.")
    phrase_tier = in_tiers[phrase_tier_name]

    text = " ".join(phrase for (begin, end, phrase) in phrase_tier)

    if args.keep_tier==[]:
        in_tiers = {} # forget all input tiers (just keeping aside phrase_tier)
    else:
        in_tiers = acmodel.praat_ifc.rename_prune_tiers(in_tiers, args.keep_tier)
        print(f"Keeping input tier(s) as {[*in_tiers.keys()]}.")

    phone_tier, word_tier = acmodel.nn_acmodel.align_wav_and_text_using_model(args.in_wav, text, model)
    sampa_phone_tier = acmodel.praat_ifc.sampify_tier(phone_tier)
    tg_dict = dict(phone=sampa_phone_tier, word=word_tier, phrase=phrase_tier)
    if args.align_tier!=[]:
        tg_dict = acmodel.praat_ifc.rename_prune_tiers(tg_dict, args.align_tier)
    print(f"Alignment goes to tier(s) {[*tg_dict.keys()]}.")

    final_tg_dict = {**in_tiers, **tg_dict} # in case of a name clash, prefer our new tiers

    acmodel.praat_ifc.unify_tier_ends(final_tg_dict, phrase_tier, max_fuzz=0.1)

    tg_txt = acmodel.praat_ifc.textgrid_file_text(final_tg_dict)

    with open(args.out_tg, 'w', encoding='utf-8') as f: # explicit utf-8 needed on Windows
        f.write(tg_txt)

    print(f"Output written to tiers {list(final_tg_dict.keys())} in \"{args.out_tg}\".")







