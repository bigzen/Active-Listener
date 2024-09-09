import torch as T
import numpy as np
import torchaudio as Ta
from scipy.io import wavfile
from transformers import Wav2Vec2FeatureExtractor as fe
#from itertools import pairwise

class MocapDataset(T.utils.data.Dataset):
    def __init__(self, head_dir: list, aud_dir: list, landM_dir: list, emo_vec: list, val_vec: list):
        #self.df = df
        self.head = head_dir
        self.aud = aud_dir
        self.landM = landM_dir
        self.emo = emo_vec
        self.val = val_vec
        self.time = 0

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        inp = self.aud[idx]
        l = self.landM[idx]
        h = self.head[idx]
        e = self.emo[idx]
        v = self.val[idx]
        X = self.readAudInp(inp)
        #wav2vec = self.readWavVec(inp)
        land = self.readLandMark(l)
        hea = self.readHeadPose(h)
        em = self.readEmotion(e)
        va = self.readValence(v)
        return X, land, hea, em, va

    def readAudInp(self, directory:list):
        waveform, sample_rate = Ta.load(directory)
        transform = Ta.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=28,
                melkwargs={"n_fft":1024, "hop_length":533, "n_mels":28, "center":False}
            )
        mfcc = transform(waveform)
        X = T.Tensor(mfcc)
        X = T.squeeze(X.permute([0,2,1]))
        X = X[:-1,:]
        self.time = X.shape[0]
        return X
    
    
    def readHeadPose(self, directory:list):
        y = []
        files = open(directory,'r')
        readlist = files.readlines()[2:]
        vals = []
        for n, line in enumerate(readlist):
            if n%4==0:
              line = line.split(' ')
              line = list(map(float,line[2:5]))
              vals.append(line)
        if len(vals)>self.time:
          diff = len(vals)-self.time                           #just commented
          vals = vals[int(diff/2):-(diff-int(diff/2))]     #just commented
        y = T.Tensor(vals)
        return y
    
    def readEmotion(self, emo:str):
        y = [emo]
        return y
    
    def readValence(self, vec:str):
        y = []
        str = vec[1:-1].split(', ')
        y = np.array(str, dtype=float)
        y = T.tensor(y, dtype=T.float)
        return y

    def readLandMark(self, directory:list):
        files = open(directory,'r')
        readlist = files.readlines()[2:]
        vals = []
        for n, line in enumerate(readlist):
            if n%4==0:
                line = line.split(' ')
                line = list(map(float,line[2:]))  #handle nan values and \n in last entry
                del line[1::3]
                line = np.nan_to_num(np.array(line[:-4]))
                vals.append(line)
        if len(vals)>self.time:
            diff = len(vals)-self.time                           #just commented
            vals = vals[int(diff/2):-(diff-int(diff/2))]     #just commented       
        y = T.Tensor(vals)
        return y

class collate_mocap(MocapDataset):
    def __init__(self, head_dir: list, aud_dir: list, landM_dir: list, emo_vec: list, val_vec: list):
        super().__init__(head_dir, aud_dir, landM_dir, emo_vec, val_vec)
    def __getitem__(self, idx):
        inp = self.aud[idx]
        l = self.landM[idx]
        h = self.head[idx]
        e = self.emo[idx]
        v = self.val[idx]
        X = self.readAudInp(inp)
        land = super().readLandMark(l)
        hea = super().readHeadPose(h)
        em = super().readEmotion(e)
        va = super().readValence(v)
        data = {'inputs':X,'labels':[hea, va]}
        return data


    
