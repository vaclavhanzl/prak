# prak: Czech phonetic alignment tool
Given a Czech voice recording and corresponding text transcript, this tool can guess a sequence
of phones and their position in time.

Design goals:
* Fully opensource, avoiding dependencies which would restrict academic and/or comercial use in 
any way. Original code was written for this project from scratch to avoid any legal limits and
allow others to continue this work.
* Usable on Linux, Mac and Windows.
* More precise than similar tools, specifically targeting needs of the Czech phoneticians.

## Instalation
First, get **prak** source code:
```
git clone https://github.com/vaclavhanzl/prak.git
```
(You can also just download a zip file - click on the green "Code" button.)

Then install prerequisities:
* [python3](https://www.python.org/)
* [PyTorch](https://pytorch.org/) with **torchaudio** (CPU version is enough, choose Conda or Pip)

That's it! If you feel scared, look down for detailed instructions.

## Usage
To get an idea about pronunciation variants being considered, run this first (being in the **prak** folder):
```
prongen/prak_prongen.py
```
The tool will nicely help you. To see generated Czech pronunciation, try this:
```
prongen/prak_prongen.py -p
```
and then type Czech sentences in terminal (finish each with Enter, press Ctrl+D to stop).
(Look at [prongen/README.md](prongen/README.md) for details.) If this works, **python** is installed OK.

For alignment with audio, you will need a Czech voice recording (wav) and a corresponding utf8 Czech plain text
transcript (in txt or TextGrid).
The transcript can contain usual interpunction (as a secretary would transcribe it) but should be precise even at
disfluencies - if the recording contains a repeated word, so should the transcript.

Multiple pronunciation variants are considered, the acoustic model will hopefully choose the right
one. You may need to teach the tool some additional foreign words or tell it about important seams in composite
words.

## Details of Linux instalation
If you just need **prak** working and do not otherwise care about pytorch, maybe this will suffice (on Debian):
```
apt-get install python3-torch
```
If you want to enjoy python and pytorch a bit more, you most likely want python's own package management and virtual environments.
For scientific work, **conda** package manager might be better than **pip**. In the conda world, there are still many options.
You likely do not want the huge Anaconda but rather the more free and modular conda forge. To get it working, you still have 
multiple options from which [**mambaforge**](https://mamba.readthedocs.io/en/latest/installation.html) (faster conda) looks quite good. With this general guidance, it is now easy to google your way.

Great part of pytorch instalation complexity stems from the CUDA GPU drivers instalation.
If you do not plan training big neural networks or do not have a decent GPU, you may very well
use pytorch just on the CPU. **Prak** only uses CPU for phone alignment and even acoustic model
can be reasonably trained on just the CPU.

## Details of Mac instalation
To get python, you may first install [brew](https://brew.sh/):
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Then you may go on with [**mambaforge**](https://mamba.readthedocs.io/en/latest/installation.html).

If you have have a Mac with the new M1 chip, you may try [Metal](https://developer.apple.com/metal/) via [MPS backend for PyTorch](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/).

## Details of Windows instalation
It *should* work but we did not try yet. (Try mambaforge?)

## About the name
Some Czech phonetitians call similar tools "nastřelovač" as these tools position phones and their time stamps
in a fast but rather unprecise manner, like positioning objects by shooting them where they should be.
The Czech word "prak" means "sling", a common shooting toy.
