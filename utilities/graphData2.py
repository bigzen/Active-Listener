import torch as T
import numpy as np
import librosa
from torch_geometric.data import Data, Dataset
class MocapDataset_pitch(Dataset):
    def __init__(self, head_dir: list, aud_dir: list, landM_dir: list, emo_vec: list, val_vec: list):
        #self.df = df
        self.head = head_dir
        self.aud = aud_dir
        self._indices = None
        self.transform = None
        self.time = 0

    def __len__(self):
        return len(self.head)
    
    def len(self):
        return len(self.head)

    def get(self, idx):
        X, edge_index = self.readAudInp(self.aud[idx])
        y = self.readHeadPose(self.head[idx])
        data = Data(x=X, edge_index = edge_index, y=y)
        return data

    def readAudInp(self, directory):
        waveform, sample_rate = librosa.load(directory)
        _, mag = librosa.piptrack(y=waveform, sr=sample_rate, n_fft=1024, hop_length=735, center=False)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=28, n_fft=1024, hop_length=735, center=False)
        pitch = np.expand_dims(np.max(mag,axis=0),axis=0)
        X = np.vstack((mfcc,pitch))
        X = T.Tensor(X)
        X = X.permute([1,0])
        self.time = X.shape[0]

        a = list(range(1,self.time))
        b = a+[0]
        c = [0]+a
        edge_index = T.Tensor(np.vstack([b,c]))
        return X, edge_index.type(dtype=T.int64)
    
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
   