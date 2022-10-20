# For Czech text in a phrase tier, make word and phone tiers, time aligned with sound.
#
# Copyright (c) 2022 Vaclav Hanzl. This is a free software (see the MIT license).
# This file is part of the https://github.com/vaclavhanzl/prak project.

#clearinfo

# Take input data from praat
snd = selected("Sound")
tg = selected("TextGrid")
tg_name$ = selected$("TextGrid")

#printline input data in

# Save it to prak's temporary directory
select 'snd'
Save as WAV file: "~/prak/tmp/tmp_phrase_align.wav"
select 'tg'
Save as text file: "~/prak/tmp/tmp_phrase_align.TextGrid"

#printline saved to temporary directory

# Run prak alignment
system ~/prak/prak -i ~/prak/tmp/tmp_phrase_align.TextGrid -w ~/prak/tmp/tmp_phrase_align.wav -o ~/prak/tmp/tmp_phrase_align.out.textgrid --force>~/prak/tmp/prak_invocation.log 2>&1

#printline prak alignment done

# Read results back to praat
Read from file: "~/prak/tmp/tmp_phrase_align.out.textgrid"
tg_out = selected("TextGrid")

#printline results read

# Replace our input textgrid (phrase, whatever else) with the output one (phone, word, phrase)
# Maybe we should avoid changing any textgrid which has other tiers than just phrase?
select 'tg'
Remove
select 'tg_out'
Rename... 'tg_name$'

printline aligned!!

