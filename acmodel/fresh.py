
# Copyright © 2023 Václav Hanžl. Part of MIT-licensed https://github.com/vaclavhanzl/prak

# Fresh NN acoustic models using all the latest modern goodies of the world
#
# Vanilla Prak tries to be easy to understand and therefore uses just a plain ReLU stack.
# In "Fresh Prak", we give up these limitations and use whatever the modern times bring.


from torch.utils.data import Dataset
import torchaudio



# Speech Dataset with vector embeddings:

class SpeechDatasetFresh(Dataset):
    def __init__(self, all_mfcc, all_targets, b_set, sideview = 9, speaker_vectors = None, all_features = None):
        """
        all_features are one-per-20ms vector embeddings of speech
        """
        self.all_mfcc = all_mfcc
        self.all_targets = all_targets
        self.all_features = all_features # vector embedings
        self.sideview = sideview
        self.speaker_vectors = speaker_vectors
        
        self.wanted_outputs = torch.eye(len(b_set), device=device).double()
        self.output_map = {}
        for i, b in enumerate(b_set):
            self.output_map[b] = self.wanted_outputs[i] # prepare outputs with one 1 at the right place
        self.ignored_end = 0

    def __len__(self):
        return len(self.all_targets) - 2*self.sideview - self.ignored_end

    def __getitem__(self, idx):
        idx += self.sideview
        mfcc_window = self.all_mfcc[idx-self.sideview:idx+self.sideview+1]

        nn_input = mfcc_window
        
        if self.speaker_vectors!=None:
            speaker_vector = self.speaker_vectors[idx]
            nn_input = torch.cat([mfcc_window, speaker_vector])
            
        nn_input = torch.cat([nn_input.flatten(), self.all_features[int(idx/2)]])
    
        return nn_input, self.output_map[self.all_targets[idx]]



def collect_training_material_fresh(hmms):
    #b_set = sorted({*"".join([hmm.b for hmm in hmms ])}) # make sorted set of all phone names in the training set
    #out_size = len(b_set)
    #in_size = hmms[0].mfcc.size(1)
    all_targets = "".join([hmm.targets for hmm in hmms])
    train_len = len(all_targets)
    all_mfcc = torch.cat([hmm.mfcc for hmm in hmms])  # .double() #.to(device)
    all_features = torch.cat([hmm.features for hmm in hmms])
    assert all_mfcc.size()[0]==train_len
    return all_mfcc.bfloat16(), all_targets, all_features.bfloat16()


def compute_wav_vector_features(hmm, bundle, dwm, n=7, device="cpu"):
    """
    Compute wav vector embedings. Example parameters:
      bundle = torchaudio.pipelines.HUBERT_BASE
      wav_model = bundle.get_model()
      dwm = wav_model.to(device)
    """
    waveform, sample_rate = torchaudio.load(hmm.wav)
    if sample_rate!=bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    waveform = waveform.double()
    features, x = dwm.extract_features(waveform.to(device))
    hmm.features = features[n][0].detach().to("cpu")


def unify_mfcc_and_features_size(hmm):
    """
    Slightly trim ends to make sure sizes do correspond
    """
    m = len(hmm.mfcc)
    f = len(hmm.features)
    #print(m, f)
    f = min(int(m/2), f)
    m = 2*f
    #print('..', m, f)
    hmm.mfcc = hmm.mfcc[:m]
    if hmm.targets != None:
        hmm.targets = hmm.targets[:m]
    hmm.features = hmm.features[:f]









