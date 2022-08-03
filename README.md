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

## About the name
Some Czech phonetitians call similar tools "nastřelovač" as these tools position phones and their time stamps
in a fast but rather unprecise manner, like positioning objects by shooting them where they should be.
The Czech word "prak" means "sling", a common shooting toy.
