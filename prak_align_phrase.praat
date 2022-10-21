### For Czech text in a phrase tier, make word and phone tiers, time aligned with sound.
###
### Copyright (c) 2022 Vaclav Hanzl & Adleta Hanzlova. This is a free software (see the MIT license).
### This file is part of the https://github.com/vaclavhanzl/prak project.

clearinfo

### Take input data from praat
snd = selected("Sound")
snd_name$ = selected$("Sound")
tg = selected("TextGrid")
tg_name$ = selected$("TextGrid")

#printline input data in

### Scale times of imput items
select 'tg'
plus 'snd'
Scale times

#printline times scaled


### Make sure textgrid is not aligned already
select 'tg'
tiercount = Get number of tiers

	for tierx from 1 to tiercount
		tiername$ = Get tier name: tierx
		if tiername$ = "phone" or tiername$ = "Phone"
			intervalcount = Get number of intervals: 1
			if intervalcount > 1
				beginPause: "phone tier clash"
					comment: "Non-empty 'phone' tier in '" + tg_name$ + "'. Overwrite it?"
				clashResponse = endPause: "Yes, continue.", "No, stop.", 2, 2
				if clashResponse = 2
					printline
					printline !!! alignment interrupted at 'tg_name$' ("phone" tier clash)
					exitScript ( )
				endif
			endif
		endif
	endfor

#printline alignment clash controll ok


### If 'phrase' tier is missing, identify the source text

if tiercount = 1
	tier1name$ = Get tier name: 1
	if not tier1name$ = "phrase"
		beginPause: "identify text"
			comment: "No 'phrase' tier in '" + tg_name$ + "'. Use '" + tier1name$ + "' as source text?"
		textSource = endPause: "Yes, use it!", "No, let me check.", 2, 2
		if textSource = 1
			Set tier name: 1, "phrase"
		endif
		if textSource = 2
			printline
			printline !!! alignment interrupted at 'tg_name$' (source text issue)
			exitScript ( )
		endif
	endif
else
	phrasetierCount = 0
	for tierz from 1 to tiercount
		tierZname$ = Get tier name: tierz
		if tierZname$ = "phrase" or tierZname$ = "Phrase"
			phrasetierCount = phrasetierCount + 1
		endif
	endfor
	if not phrasetierCount = 1
		printline identify source text in 'tg_name$'
		for tierw from 1 to tiercount
			tierWname$ = Get tier name: tierw
			printline tier 'tierw': 'tierWname$'
		endfor
		if phrasetierCount = 0
			beginPause: "identify text"
				comment: "No 'phrase' tier in '" + tg_name$ + "'. Which tier is source text?"
				natural: "source text tier", tiercount
			textSource = endPause: "Continue", "What? Stop alignment!", 1, 2
			if textSource = 1
				Set tier name: source_text_tier, "phrase"
			endif
		elsif phrasetierCount > 1
			beginPause: "identify text"
				comment: "Multiple 'phrase' tiers in '" + tg_name$ + "'. Which tier is source text?"		
				natural: "source text tier", tiercount
			textSource = endPause: "Continue", "What?? Stop alignment!", 1, 2
			if textSource = 1
				Set tier name: source_text_tier, "phrase"
				tg_rm = tg
				tg = Extract one tier: source_text_tier
				select 'tg_rm'
				Remove
				select 'tg'
			endif
		endif
		if textSource = 2
			printline
			printline !!! alignment interrupted at 'tg_name$' (source text issue)
			exitScript ( )
		endif
	endif
endif

#printline source text ok


### Save it to prak's temporary directory
select 'snd'
Save as WAV file: "~/prak/tmp/tmp_phrase_align.wav"
select 'tg'
Save as text file: "~/prak/tmp/tmp_phrase_align.TextGrid"

#printline saved to temporary directory


### Run prak alignment
system ~/prak/prak -i ~/prak/tmp/tmp_phrase_align.TextGrid -w ~/prak/tmp/tmp_phrase_align.wav -o ~/prak/tmp/tmp_phrase_align.out.textgrid --force>~/prak/tmp/prak_invocation.log 2>&1

#printline prak alignment done


### Read results back to praat
Read from file: "~/prak/tmp/tmp_phrase_align.out.textgrid"
tg_out = selected("TextGrid")

#printline results read


### Replace our input textgrid (phrase, whatever else) with the output one (phone, word, phrase)
select 'tg'
Remove
select 'tg_out'
Rename... 'snd_name$'

printline aligned!!

