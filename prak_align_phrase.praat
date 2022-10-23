### For Czech text in a phrase tier, make word and phone tiers, time aligned with sound.
###
### Copyright (c) 2022 Vaclav Hanzl & Adleta Hanzlova. This is a free software (see the MIT license).
### This file is part of the https://github.com/vaclavhanzl/prak project.

### Config section - where is prak, mambaforge and exceptions.txt
if windows
       prak$ = "C:\prak"
       mambaforge$ = "C:\mambaforge"
       #prak$ = "C:\Users\Ferda\prak"
       #mambaforge$ = "C:\Users\Ferda\mambaforge"
       exceptions$ = prak$ + "\exceptions.txt"
else
       prak$ = "~/prak"
       mambaforge$ = "~/mambaforge"
       exceptions$ = prak$ + "/exceptions.txt"
endif


clearinfo

### Compute all the remaining system interaction components from config variables
if windows
       prak_tmp$ = prak$ + "\tmp\"
       python$ = mambaforge$ + "\python.exe"
       prak_py$ = prak$ + "\prak.py"
       prak_exe$ = python$ + " " + prak_py$
else
       prak_tmp$ = prak$ + "/tmp/"
       prak_exe$ = prak$ + "/prak"
endif
prak_tgin$ = prak_tmp$ + "tmp_phrase_align.TextGrid"
prak_wav$ = prak_tmp$ + "tmp_phrase_align.wav"
prak_tgout$ = prak_tmp$ + "tmp_phrase_align.out.TextGrid"
prak_log$ = prak_tmp$ + "prak_invocation.log"
#printline Will invoke prak like this:
#printline 'prak_exe$' -i 'prak_tgin$' -w 'prak_wav$' -e 'exceptions$' -o 'prak_tgout$' --force >'prak_log$' 2>&1


### Assess input data
tg_inputCount = numberOfSelected ("TextGrid")
snd_inputCount = numberOfSelected ("Sound")

for t from 1 to tg_inputCount
	tgInput't' = selected ("TextGrid",t)
endfor

for s from 1 to snd_inputCount
	sndInput's' = selected ("Sound",s)
endfor

reuse_tg = 0

### Prevent item count mismatch / decide on source text usage

if not tg_inputCount = snd_inputCount

	if tg_inputCount = 1
		select tgInput1
		tgtoReuse$ = selected$ ("TextGrid")
		beginPause: "item count mismatch"
			comment: "Use textgrid '" + tgtoReuse$ + "' to align " + string$(snd_inputCount) + " sounds?"
		reuse_tg = endPause: "Yes please!", "No, stop alignment.", 1, 2
		if reuse_tg = 2
			printline
			printline !!! alignment interrupted (item mismatch)
			exitScript ( )
		endif
	else
		beginPause: "item count mismatch"
			comment: "Cannot align " + string$(snd_inputCount) + " sounds using " + string$(tg_inputCount) + " textgrids."
		mismatch = endPause: "Ok, I'll fix it.", 1, 1
		printline
		printline !!! alignment interrupted (item count mismatch)
		exitScript ( )
	endif
endif

#printline item mismatch controll ok

### Script repetition for multiple alignments (endfor at the end of script)

for runNum from 1 to snd_inputCount
	if reuse_tg = 1
		tgInput=tgInput1
	else
		tgInput=tgInput'runNum'
	endif
	sndInput=sndInput'runNum'
	select tgInput
	plus sndInput


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
			if reuse_tg = 0
				if intervalcount > 1
					beginPause: "phone tier clash"
						comment: "Non-empty 'phone' tier in '" + tg_name$ + "'. Overwrite it?"
					clashResponse = endPause: "Yes, continue.", "No, stop.", 2, 2
					if clashResponse = 2
						printline
						printline !!! alignment interrupted at 'snd_name$' ("phone" tier clash)
						exitScript ( )
					endif
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
			printline !!! alignment interrupted at 'snd_name$' (source text issue)
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
				tgInput1 = tg
			endif
		endif
		if textSource = 2
			printline
			printline !!! alignment interrupted at 'snd_name$' (source text issue)
			exitScript ( )
		endif
	endif
endif

#printline source text ok


### Save it to prak's temporary directory
select 'snd'
Save as WAV file: prak_wav$
select 'tg'
Save as text file: prak_tgin$

#printline saved to temporary directory


### Run prak alignment
system 'prak_exe$' -i 'prak_tgin$' -w 'prak_wav$' -e 'exceptions$' -o 'prak_tgout$' --force >'prak_log$' 2>&1

#printline prak alignment done


### Read results back to praat
Read from file: prak_tgout$
tg_out = selected("TextGrid")

#printline results read


### Replace our input textgrid (phrase, whatever else) with the output one (phone, word, phrase)
if reuse_tg = 0
	select 'tg'
	Remove
endif
select 'tg_out'
Rename... 'snd_name$'

### End of alignment repetition
endfor

printline aligned!!

