# Specialized plot functions for work on the acoustic model

import matplotlib.pyplot as plt
#%matplotlib ipympl
import matplotlib as mpl

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import torch
import torchaudio

mpl.rcParams['figure.figsize'] = (20,3)
# Use full with and waste less space on prompts on the left:
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
display(HTML("<style>.prompt_container{width: 11ex !important; }</style>"))
display(HTML("<style>div.prompt{min-width: 11ex; }</style>"))

cmap = ['BrBG', 'RdBu', 'PuOr'][0]
def plot_specgram(waveform, sample_rate):
    waveform = waveform.numpy()
    figure = plt.figure()
    plt.specgram(waveform[0], Fs=sample_rate, cmap=cmap)
    plt.tick_params(axis="x",direction="in", pad=-12)
    plt.tick_params(axis="y",direction="in", pad=-38)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show(block=False)
    figure.canvas.header_visible = False
    figure.canvas.footer_visible = False
    #plt.clf()

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)
    #axes = plt
    axes.tick_params(axis="x",direction="in", pad=-12)
    axes.tick_params(axis="y",direction="in", pad=-22)
    axes.plot(time_axis, waveform[0], linewidth=1, color=(0,0.4,0.4))
    axes.grid(True)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show(block=False)
    max_time = num_frames/sample_rate
    axes.set_xlim([0,max_time])
    figure.canvas.header_visible = False
    figure.canvas.footer_visible = False
    #plt.clf()

def plot_wavfile(wavfile):
    waveform, fs = torchaudio.load(wavfile)
    plot_waveform(waveform, fs)

def plot_fun(x, frame_sample_rate=100):
    #figure = plt.figure()
    figure, axes = plt.subplots(1, 1)
    plt.plot(x)
    plt.tick_params(axis="x",direction="in", pad=-12)
    plt.tick_params(axis="y",direction="in", pad=-38)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    num_frames = x.size()[0]
    max_time = num_frames/frame_sample_rate
    max_time = num_frames
    axes.set_xlim([0,max_time])
    axes.grid(True)
    plt.show(block=False)
    figure.canvas.header_visible = False
    figure.canvas.footer_visible = False


def plot_matrix(x, cmap='PuOr', figsize=(20,3), footer=False):
    """
    Plot matrix as image. Setting footer=True activates measurement
    of mouse coordinates in the matrix.
    """
    figure = plt.figure(figsize=figsize)
    plt.imshow(x.T, cmap=cmap)
    plt.tick_params(axis="x",direction="in", pad=-12)
    plt.tick_params(axis="y",direction="in", pad=-38)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show(block=False)
    figure.canvas.header_visible = False
    if not footer:
        figure.canvas.footer_visible = False


