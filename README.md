# prak: Czech phonetic alignment tool
Given a Czech voice recording and corresponding text transcript, this tool can guess a sequence
of phones and their position in time.

Design goals:
* Fully opensource, avoiding dependencies which would restrict academic and/or comercial use in 
any way. Original code was written for this project from scratch to avoid any legal limits and
allow others co continue this work.
* Usable on Linux, Mac and Windows.
* More precise than similar tools, specifically targeting needs of the Czech phoneticians.

## Instalation
Prerequisities:
* python

For in-depth experiments and development, you may also want:
* praat
* kaldi
* CommonVoice Czech data

## Usage
To get an idea about pronunciation variants being considered, run this first:
```
prongen/prak_prongen.py
```
The tool will nicely help you.

## How pronunciation generator works
This tool was made to help phoneticians and therefore it tries to capture any logic (as far as it exists)
in the Czech pronunciation and really make it right. There are more pragmatic approaches, for example one
may generate long list of examples and train one of the proven pronunciation generation tools on these,
or write some 350 search/replace rules and solve pronunciation with these, or just make long enough
pronunciation lexicon by hand. Any of these is sufficient to make a good ASR because e.g. triphone models
can absorb and fix significant errors in the pronunciation layer and modern NN systems even ignore the
concept of anything like phones or phonemes.

With this tool, we took the painful way of searching for logic in the language. To a certain degree, this
works for things like Czech voiced/voiceless assimilations and can even do a better job than
autolearned rules as the assimilation dependencies can be quite long range sometimes. For pronunciation
of "slightly foreign" words, we made list of examples and automatically learned good rule set for
pronunciation of ditini/dytyny. And for the rest, we tried to make it easy for the user to add hand-made
rules.

As this tool was designed specifically for time alignment, we preffered generating multiple pronunciation
variants whenever there is even a slight chance for these - acoustic model can select among variants.
On the other hands, we tried to allow variants only where it makes sense - for example, at the meeting
point of two words, there are multiple possibilities for voiced/voiceless variant of word's end, there
may or may not be an intervening space and the second word could possibly start with a glottal stop
(or not) if it starts with a vowel. Not all combinations make sense and we try to allow only those which
do. By weeding out the nonsense combinations, we hope to create a more capable acoustic models even for
problematic elements like the glottal stop or voiced alophone of the Czech "ch". (These things have no
sure occurences marked in the Czech ortography so we must carefully guide the learning process for these.)

The overall structure of the pronunciation variants generator is roughly as follows:
* clean up punctation marks
* lowercase the text
* connect short prepositions to words
* treat foreign words using hand-made string replacement rules and learned (on hand-made examples) rules for ditini/dytyny, also guess some important seams in composed words
* care for few forward assimilations (mostly for "ř" and initial "sh")
* do bulk of the "logical" work in a system of backward assimilations (each of the parts is in fact a Finite State Transducer with a very limited state set - backward direction makes these FSTs much simpler or even possible at all):
  - expansion of "multiphone" h/ch created in the forward assimilation step above
  - glottal stop insertion
  - voiced/voiceless assimilations
  - dtn/ďťň processing in groups like "ntní", processing of bě/pě/vě/mě/fě
  - velarization in nk/ng
* optionally merge some phones (e.g. doubled ones) by coarticulation rules. (For technical reasons, this step is in fact done before the preceding "logical" step.)
* optimize the resulting graph of possibilities so as it can be presented in a human readable way

## Example output
```
prak/prongen$ ./prak_prongen.py -p
--- reading stdin, printing possible pronunciations ---

slyšel jsem asi galantní zpjěv ptáků

           |ʔ
           _ʔ
slıšel .sem_ ası galantňi: spjef pta:ku:
       j             nť
                     ňť


v USA pneumatiky za eura nekoupíš

      ʔ         eu
f=ʔu:=.es=.a: pne͡umatıkı za_ e͡ura neko͡upi:š
v=        ʔ                _ʔ
                           |ʔ


to jsou úřední dokumenty

       |ʔ
       _ʔ
to .so͡u_ u:ředňi: dokumentı
   j         ď


nashromáždilo se to na tom se shodneme už dnes                     
                     
                                zh         ž_
na=sxroma:žďılo se to na tom se sxodneme_ uš|dnes
   zh                                   _ʔ
                                        |ʔ         
```

## About the name
Some Czech phonetitians call similar tools "nastřelovač" as these tools position phones and their time stamps
in a fast but rather unprecise manner, like positioning objects by shooting them where they should be.
The Czech word "prak" means "sling", a common shooting toy.
