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

## Installation on Windows
Go to [mambaforge](https://mamba.readthedocs.io/en/latest/installation.html), download mambaforge for windows ([Mambaforge-Windows-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe)). Run it, despite any protests from Windows. Install to ```C:\mambaforge```. This location is the only thing you have to change, otherwise go with default settings in installation dialogs.

Run mambaforge console. Prompt must start ```(base)```. (If it does not, you run something else. Find mambaforge console.) Type this:
```
mamba install pytorch torchaudio -c pytorch
pip install pysoundfile
```

Download prak v1 zip file from this GitHub page - change "main" to "v1" up left, click on the green "Code" button up right, download zip. Uncompress it directly to ```C:\```, creating ```C:\prak```.

Run praat. Open script ```C:\prak\prak_align_phrase.praat``` and add it to menu (File > Add to dynamic menu > Class1: Sound, Class2: TextGrid, Command: Align using prak). In preferences, set text files read/write encoding to UTF-8.

Refer to [Prak installation details](https://github.com/vaclavhanzl/prak/wiki/Prak-installation-details) if you need to know more.

## Installation on Mac
Go to [mambaforge](https://mamba.readthedocs.io/en/latest/installation.html).
Choose and download the installation file [here](https://github.com/conda-forge/miniforge#mambaforge), either
[Mambaforge-MacOSX-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-x86_64.sh)
for older Intel-based Macs or
[Mambaforge-MacOSX-arm64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-arm64.sh)
for new Macs with Apple M1 chips. Then make the downloaded file executable (using chmod) and run it in the terminal, for example (for Intel Mac):
```
chmod +x Downloads/Mambaforge-MacOSX-x86_64.sh 
Downloads/Mambaforge-MacOSX-x86_64.sh
```
Answer a few questions (likely **yes** or just Enter but you must explicitly type **yes** where needed). If the licence agreement is too long to show, press spacebar untill you reach the end, then agree (yes). After installing mambaforge successfully, QUIT TERMINAL COMPLETELY and run it again. The prompt will now start with "(base)" and you can install python packages we need:
```
mamba install pytorch torchaudio -c pytorch
```

Get **prak**, preferably using **git** (makes future updates super easy):
```
git clone https://github.com/vaclavhanzl/prak.git
cd prak
git checkout v1
```
(Or you can get zip as described above for Windows and uncompress it in your home folder.)

Run praat. Open script ```C:\prak\prak_align_phrase.praat``` and add it to menu (File > Add to dynamic menu > Class1: Sound, Class2: TextGrid, Command: Align using prak). Set read/write to UTF-8.

Refer to [Prak installation details](https://github.com/vaclavhanzl/prak/wiki/Prak-installation-details) if you need to know more.


## Installation on Linux
Things happen too fast in the python world for apt package managers to keep up. So you most likely want a special package manager just for python, and there are really great tools to choose from.

For scientific work, **conda** package manager might be better than **pip**. In the conda world, there are still many options.
You likely do not want the huge Anaconda but rather the more free and modular conda forge. To get it working, you still have 
multiple options from which [**mambaforge**](https://mamba.readthedocs.io/en/latest/installation.html) (faster conda) looks quite good. With this general guidance, it is now easy to google your way.

Big part of pytorch installation complexity stems from the CUDA GPU drivers installation.
If you do not plan training big neural networks or do not have a decent GPU, you may very well
use pytorch just on the CPU. **Prak** only uses CPU for phone alignment and even acoustic model
can be reasonably trained on just the CPU.

You need ```pytorch``` and ```torchaudio``` packages. As a linux hacker you likely have other things to do with python so **prak** tries to use its own virtual environment ```prak``` before resorting to ```base``` or even trying to run without a virtual environment
(this logic is in [prak/prak](https://github.com/vaclavhanzl/prak/blob/main/prak)). Prepare it like this:
```
mamba create prak
mamba activate prak
mamba install pytorch torchaudio -c pytorch
```

Get **prak** source code via git. If you have a GitHub account and ssh key, clone directly from your home like this:
```
git clone git@github.com:vaclavhanzl/prak.git
cd prak
git checkout v1
```
otherwise resort to https:
```
git clone https://github.com/vaclavhanzl/prak.git
cd prak
git checkout v1
```

Then run **praat** and add ```prak/prak_align_phrase.praat``` to menu, the same way as described above for Mac.
If you did not put **mamba** or **prak** directly to your home directory, edit config variables in this script.

Refer to [Prak installation details](https://github.com/vaclavhanzl/prak/wiki/Prak-installation-details) for even more details.

## Usage
### Align from praat GUI
Select sound and textgrid containing phrase tier, press your new **Align using prak** button.

You can also align multiple sounds using multiple textgrids, or multiple sounds using one input textgrid for all.

### Align from command line
For alignment with audio, you will need a Czech voice recording (wav) and a corresponding Czech transcript (in phrase tier in an utf8 TextGrid).
The transcript can contain usual punctuation (as a secretary would transcribe it) but should be precise even at
disfluencies - if the recording contains a repeated word, so should the transcript. If you used **praat** to
make a phrase tier with transcript and saved it to file ```pocasi.textgrid```, you can use **prak** to guess
pronunciations and time align all phones and words. Run this in terminal:
```
~/prak/prak.py -i pocasi.textgrid -w pocasi.wav -o pocasi.out.textgrid
```
The **Align using prak** button is in fact using this method internally to do its magic.

### Generate just pronunciations
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


Multiple pronunciation variants are considered, the acoustic model will hopefully choose the right
one. You may need to teach the tool some additional foreign words or tell it about important seams in composite
words. Just edit the [prak/exceptions.txt file](https://github.com/vaclavhanzl/prak/blob/main/exceptions.txt).











## Common details of installation for all the platforms
You need to install these prerequisities:
* [python3](https://www.python.org/)
* [PyTorch](https://pytorch.org/) with **torchaudio** (CPU version is enough, choose Conda or Pip)

There are many ways to do it and the sites above document them very well. But you may just follow our step-by-step
guides below.



## Prak updates
If you used **git** to download **prak**, it is easy. Go to the **prak** folder and run:
```
git pull
```
(If you went with the zip option, download a new zip the same way.)

## Speed
1 minute audio was phone-aligned in 18 seconds on a 2014 Mac Air, in 3.6 seconds on a 2020 M1 Mac Air, in 5 seconds on a decent 2020 Intel linux box.

## Training data
The acoustic model was trained solely on the Czech [CommonVoice](https://commonvoice.mozilla.org/en/datasets) recordings.
I am also immensly grateful to [Fonetický ústav FFUK](https://fonetika.ff.cuni.cz/) for the possibility to compare
their hand labeled (and phone-level time aligned) data with prak's automatic labeling, this helps a lot. However, training
of the prak's acoustic model currently does not involve any hand-aligned data. (CommonVoice data consist of recorded sentences with
corresponding ortographic transcriprion. No human-decided time alignment on the level of words or phones is used for training,
these alignments are purely machine made.)

## About the name
Some Czech phonetitians call similar tools "nastřelovač" as these tools position phones and their time stamps
in a fast but rather unprecise manner, like positioning objects by shooting them where they should be.
The Czech word "prak" means "sling", a common shooting toy.

## Work in progress
We are currently testing and tuning installation for all the platforms so this page is still changing quickly.

The main branch has all the code but NOT the trained acoustic model. We still search for a good and easy way to
deliver it to you. Currently the model is added in the **v1** branch. So do not forget:
```
git checkout v1
```
(This trickery should avoid cumulation of old models in prak git repo. Branch can be deleted on github. Commit in main cannot.)

## Can I contribute to prak?
Of yourse! You are welcome! You can contribute in many ways:
* Try to install and use prak and [report any issues](https://github.com/vaclavhanzl/prak/wiki/Reporting-bugs), e.g. in [Discussions](https://github.com/vaclavhanzl/prak/discussions)
* [Become a developer](https://github.com/vaclavhanzl/prak/wiki/Developer's-guide)
* Add pronunciations of foreign words or help for harder Czech words, like composed Czech words with vowel after seam.


## Discussions and Contact
You can discuss prak in public here in [Discussions](https://github.com/vaclavhanzl/prak/discussions).
If you want to tell me more personally that you love this or hate this, message @vaclav512 on Twitter.

## Thanks
I'd like to thank all the nice people from [Fonetický ústav FFUK](https://fonetika.ff.cuni.cz/) and equally nice young people from [FňUK](https://www.facebook.com/profile.php?id=100057425272524) who inspired me by their charming approach to science and life.
