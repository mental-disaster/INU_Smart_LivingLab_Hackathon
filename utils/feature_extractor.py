import os
import librosa
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.config import get_config

config = get_config()

class FEATURE_EXTRACTOR():
    def __init__(self):
        self.sampling_rate = config.sr
        self.n_fft = config.n_fft
        self.filter = config.filter
        self.mfc_dim = config.mfc
        self.hop_length = config.hop_len
        self.win_length = config.win_len

        
    def get_mel(self, file):
        S, _ = librosa.load(file, sr=self.sampling_rate)
        mel = librosa.feature.melspectrogram(S,
                                             sr=self.sampling_rate,
                                             n_fft=self.n_fft,
                                             n_mels=self.mel_dim,
                                             hop_length=self.hop_length,
                                             win_length=self.win_length)
        feature = librosa.power_to_db(mel, ref=np.max)

        return feature
    
    def get_mfcc(self, file):
        S, _ = librosa.load(file, sr=self.sampling_rate)
        mel = librosa.feature.melspectrogram(S,
                                             sr=self.sampling_rate,
                                             n_fft=self.n_fft,
                                             n_mels=self.filter,
                                             hop_length=self.hop_length,
                                             win_length=self.win_length)
        log_S = librosa.power_to_db(mel, ref=np.max)

        mfcc = librosa.feature.mfcc(S=log_S,
                                    n_mfcc=self.mfc_dim,
                                    sr=self.sampling_rate,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    win_length=self.win_length
                                    )
        mfcc_delta = librosa.feature.delta(mfcc, width=3)
        mfcc_delta2 = librosa.feature.delta(mfcc, width=3, order=2)

        feature = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)

        return feature


if __name__ == "__main__":
    # Train data
    wavPath = config.data_path + '/train'
    featPath = config.feat_path + '/train'
    ctrlPath = config.ctrl

    with open(ctrlPath, 'r') as f:
        files = f.read()
    ctrl = files.splitlines()

    extractor = FEATURE_EXTRACTOR()
    os.makedirs(featPath, exist_ok=True)

    for line in ctrl:
        inputPath = os.path.join(wavPath, line)
        outputPath = os.path.join(featPath, line)

        mfc_feat = extractor.get_mfcc(inputPath + '.wav')
        np.savetxt(outputPath + '.mfc', mfc_feat)


    # Test data
    wavPath = config.data_path + '/eval'
    featPath = config.feat_path + '/eval'
    ctrlPath = config.ctrl.replace('train', 'eval')

    with open(ctrlPath, 'r') as f:
        files = f.read()
    ctrl = files.splitlines()

    os.makedirs(featPath, exist_ok=True)

    for line in ctrl:
        inputPath = os.path.join(wavPath, line)
        outputPath = os.path.join(featPath, line)

        mfc_feat = extractor.get_mfcc(inputPath + '.wav')
        np.savetxt(outputPath + '.mfc', mfc_feat)