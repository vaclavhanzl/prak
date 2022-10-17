# prak: Czech phonetic alignment tool

## I just want it, now!
OK. Get **prak** source code and python3.10 with pytorch and torchaudio. Record wav, make phrase
tier with text and **prak** will find phone and word times: 
```
prak -i pocasi.textgrid -w pocasi.wav -o pocasi.out.textgrid
```
![example phone tier](prongen/doc/images/tiers_example.png)
Enjoy!
## Well, not so fast. Explain it all nicely.
Given a Czech voice recording and corresponding text transcript, this tool can guess a sequence
of phones and their position in time.

Design goals:
* Fully opensource, avoiding dependencies which would restrict academic and/or comercial use in 
any way. Original code was written for this project from scratch to avoid any legal limits and
allow others to continue this work.
* Usable on Linux, Mac and Windows.
* More precise than similar tools, specifically targeting needs of the Czech phoneticians.

## Installation
First, get **prak** source code:
```
git clone https://github.com/vaclavhanzl/prak.git
```
Or you can instead download a zip file - click on the green "Code" button (up and right on this page).

Than you need to install python3.10, pytorch and torchaudio, e.g. using
[mambaforge](https://mamba.readthedocs.io/en/latest/installation.html).

If you feel scared, look down for detailed instructions.

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
The transcript can contain usual punctuation (as a secretary would transcribe it) but should be precise even at
disfluencies - if the recording contains a repeated word, so should the transcript. If you used **praat** to
make a phrase tier with transcript and saved it to file ```pocasi.textgrid```, you can use **prak** to guess
pronunciations and time align all phones and words. Run this in terminal:
```
prak -i pocasi.textgrid -w pocasi.wav -o pocasi.out.textgrid
```
Multiple pronunciation variants are considered, the acoustic model will hopefully choose the right
one. You may need to teach the tool some additional foreign words or tell it about important seams in composite
words. Using a simple binding praat script, you can also do all this directly from the **praat** GUI.

## Common details of installation for all the platforms
You need to install these prerequisities:
* [python3](https://www.python.org/)
* [PyTorch](https://pytorch.org/) with **torchaudio** (CPU version is enough, choose Conda or Pip)

There are many ways to do it and the sites above document them very well. But you may just follow our step-by-step
guides below.
## Details of Linux installation
If you just need **prak** working and do not otherwise care about pytorch, maybe this could be a good start (on Debian):
```
apt-get install python3-torch
```
but you still need **torchaudio** which likely is not available this way, so you may as well directly try the way described below.

If you want to enjoy python and pytorch a bit more, you most likely want python's own package management and virtual environments.
For scientific work, **conda** package manager might be better than **pip**. In the conda world, there are still many options.
You likely do not want the huge Anaconda but rather the more free and modular conda forge. To get it working, you still have 
multiple options from which [**mambaforge**](https://mamba.readthedocs.io/en/latest/installation.html) (faster conda) looks quite good. With this general guidance, it is now easy to google your way.

Big part of pytorch installation complexity stems from the CUDA GPU drivers installation.
If you do not plan training big neural networks or do not have a decent GPU, you may very well
use pytorch just on the CPU. **Prak** only uses CPU for phone alignment and even acoustic model
can be reasonably trained on just the CPU.

## Details of Mac installation
Good start is [mambaforge](https://mamba.readthedocs.io/en/latest/installation.html).
Choose and download the installation file [here](https://github.com/conda-forge/miniforge#mambaforge), either
[Mambaforge-MacOSX-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-x86_64.sh)
for older Intel-based Macs or
[Mambaforge-MacOSX-arm64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-arm64.sh)
for new Macs with Apple M1 chips. Then make the downloaded file executable and run it in the terminal, for example (for Intel Mac):
```
chmod +x Downloads/Mambaforge-MacOSX-x86_64.sh 
Downloads/Mambaforge-MacOSX-x86_64.sh
```
Answer a few questions (likely **yes** or just Enter but you must explicitly type **yes** where needed). Then QUIT TERMINAL COMPLETELY and run it again. The prompt will now start with "(base)" and you can install python packages we need:
```
mamba install pytorch torchaudio -c pytorch
```
Than you can verify that packages are really available. Run python:
```
python3
```
and at the ">>>" prompt type:
```
import torch
import torchaudio
```
If no errors appear, you won! Quit python with Ctrl+D and try **prak**.

#### Alternative ways of Mac installation
To get python, you may first install [brew](https://brew.sh/):
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Then you may go on with [**mambaforge**](https://mamba.readthedocs.io/en/latest/installation.html):
```
brew install --cask micromamba
```
But when you try to use **micromamba**, you will likely get complaints about unidentified developer (you would have to loosen security settings to get over this).

If you have have a Mac with the new M1 chip, you may try [Metal](https://developer.apple.com/metal/) via [MPS backend for PyTorch](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/).

## Details of Windows installation
It *should* work but we did not try yet. (Try mambaforge?)

## Prak updates
If you used **git** to download **prak**, it is easy. Go to the **prak** folder and run:
```
git pull
```
(If you went with the zip option, download a new zip the same way.)

## Speed
1 minute audio was phone-aligned in 18 seconds on a 2014 Mac Air, in 5 seconds on a decent 2020 Intel linux box.

## About the name
Some Czech phonetitians call similar tools "nastřelovač" as these tools position phones and their time stamps
in a fast but rather unprecise manner, like positioning objects by shooting them where they should be.
The Czech word "prak" means "sling", a common shooting toy.

## Work in progress
We are currently testing and tuning installation for all the platforms so this page is still changing quickly.
The main branch in git has working pronunciation generator. For the acoustic alignment, you currently have to resort
to the develop branch:
```
git checkout develop
```
and manually add the trained neural model, which is not in the repo. Just ask Vaclav Hanzl :) We will certainly
improve these deployment details soon...

## Discussions and Contact
You can discuss hacktrack in public here in [Discussions](https://github.com/vaclavhanzl/prak/discussions).
If you want to tell me more personally that you love this or hate this, message @vaclav512 on Twitter.

## Thanks
I'd like to thank all the nice people from [Fonetický ústav FFUK](https://fonetika.ff.cuni.cz/) and equally nice young people from [FňUK](https://www.facebook.com/profile.php?id=100057425272524) who inspired me by their charming approach to science and life.
