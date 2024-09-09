import torch as T
import numpy as np
import opensmile
import librosa
import audiofile
import torchaudio as Ta
from transformers import Wav2Vec2Tokenizer as fe
from transformers import Wav2Vec2Model as w2v2
from torch_geometric.data import Data, Dataset
#from itertools import pairwise

class MocapDataset(Dataset):              #this class get cobines wav2vec2 with valence
    #This data set is for rate of change prediction
    def __init__(self, head_dir: list, aud_dir: list, landM_dir: list, emo_vec: list, val_vec:list):
        #self.df = df
        self.head = head_dir
        self.aud = aud_dir
        self.landM = landM_dir
        self.emo = emo_vec
        self.val = val_vec
        self.time = 0
        self._indices = None
        self.transform = None
        model_name = 'facebook/wav2vec2-base-960h'#"facebook/wav2vec2-large-xlsr-53"
        self.fextractor = fe.from_pretrained(model_name)
        self.model = w2v2.from_pretrained(model_name)

    def __len__(self):
        return len(self.head)
    
    def len(self):
        return len(self.head)
    
    def get(self, idx):
        X, edge_index = self.readAudInp(self.aud[idx])
        y = self.readHeadPose(self.head[idx])
        #em = self.readValence(self.val[idx])                #comment to remove val association
        #em = em.unsqueeze(0).expand(X.shape[0],-1)          #comment to remove val association
        #print(X.shape, em.shape)
        #X = T.cat((X,em[:,:-1]),dim=1)                      #comment to remove val association
        #print(X.shape)
        data = Data(x=X, edge_index = edge_index, y=y)
        return data

    def readAudInp(self, directory:list):
        data, samp = Ta.load(directory)
        input = self.fextractor(data, return_tensors='pt', sampling_rate=samp)
        with T.no_grad():
            inp = input.input_values.squeeze(0)
            feature = self.model(inp)
        feat = feature.extract_features.squeeze()
        self.time = feat.shape[0]
        a=list(range(1,self.time))
        b = a+[0]
        c = [0]+a
        edge_index = T.Tensor(np.vstack([b,c]))
        return feat, edge_index.type(dtype=T.int64)
    
    def readValence(self, vec:str):
        y = []
        str = vec[1:-1].split(', ')
        y = np.array(str, dtype=float)
        y = T.tensor(y, dtype=T.float)
        return y


    def readHeadPose(self, directory:list):
        y = []
        files = open(directory,'r')
        readlist = files.readlines()[2:]
        vals = []
        for n, line in enumerate(readlist):
            if n%2==0:
                line = line.split(' ')
                line = list(map(float,line[2:5]))
                vals.append(line)
        if len(vals)>self.time:
            diff = len(vals)-self.time                           #just commented
            vals = vals[int(diff/2):-(diff-int(diff/2))]     #just commented
        if len(vals)<self.time:
            diff = self.time-len(vals)
            for i in range(diff):
                if i%2==0:
                    vals.insert(0,vals[0])
                else:
                    vals.append(vals[-1])
        y = T.Tensor(vals)
        return y

    def readLandMark(self, directory:list):
        files = open(directory,'r')
        readlist = files.readlines()[2:]
        vals = []
        start = []
        for n, line in enumerate(readlist):
            if n==0:
                line = line.split(' ')
                line = list(map(float,line[2:]))  #handle nan values and \n in last entry
                del line[1::3]
                line = np.nan_to_num(np.array(line[:-4]))
                line0 = line
                start = line
                continue
            if n%2==0:
              line = line.split(' ')
              line = list(map(float,line[2:]))  #handle nan values and \n in last entry
              del line[1::3]
              line = np.nan_to_num(np.array(line[:-4]))
              res = list(line-line0)
              vals.append(res)
              line0=line
        if len(vals)>self.time:
          diff = len(vals)-self.time                           #just commented
          vals = vals[int(diff/2):-(diff-int(diff/2))]     #just commented       
        if len(vals)<self.time:
            diff = self.time-len(vals)
            for i in range(diff):
                if i%2==0:
                    vals.insert(0,vals[0])
                else:
                    vals.append(vals[-1])
        y = T.Tensor(vals)
        return y, start
    
class MocapDataset_mfc(Dataset):
    def __init__(self, head_dir: list, aud_dir: list, landM_dir: list, emo_vec: list, val_vec: list):
        #self.df = df
        self.head = head_dir
        self.aud = aud_dir
        self.landM = landM_dir
        self.emo = emo_vec
        self._indices = None
        self.transform = None
        self.val = val_vec
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

class MocapDataset_eg(Dataset):
    #This data set is for rate of change prediction
    def __init__(self, head_dir: list, aud_dir: list, landM_dir: list, emo_vec: list, val_vec:list):
        #self.df = df
        self.head = head_dir
        self.aud = aud_dir
        self.landM = landM_dir
        self.emo = emo_vec
        self.val = val_vec
        self.time = 0
        self._indices = None
        self.transform = None
        #model_name = 'facebook/wav2vec2-base-960h'#"facebook/wav2vec2-large-xlsr-53"
        #self.fextractor = fe.from_pretrained(model_name)
        #self.model = w2v2.from_pretrained(model_name)

    def __len__(self):
        return len(self.head)
    
    def len(self):
        return len(self.head)
    
    def get(self, idx):
        inp = self.aud[idx]
        out = self.head[idx]
        X, edge_index = self.readAudInp(inp)
        y = self.readHeadPose(out)
        data = Data(x=X, edge_index = edge_index, y=y)
        return data

    def readAudInp(self, directory:list):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
            )
        signal, sampling_rate = audiofile.read(directory)
        out = []
        for i in range(0,len(signal)-1, int(sampling_rate*0.034)):
            out.append(smile.process_signal(signal[i:i+int(sampling_rate*0.64)],sampling_rate))
        out = np.concatenate(out,0)
        out = np.nan_to_num(out)
        X = T.Tensor(out)
        self.time = X.shape[0]

        a=list(range(1,self.time))
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
            if n%2==0:
                line = line.split(' ')
                line = list(map(float,line[2:5]))
                vals.append(line)
        if len(vals)>self.time:
            diff = len(vals)-self.time                           #just commented
            vals = vals[int(diff/2):-(diff-int(diff/2))]     #just commented
        if len(vals)<self.time:
            diff = self.time-len(vals)
            for i in range(diff):
                if i%2==0:
                    vals.insert(0,vals[0])
                else:
                    vals.append(vals[-1])
        y = T.Tensor(vals)
        return y

    def readLandMark(self, directory:list):
        files = open(directory,'r')
        readlist = files.readlines()[2:]
        vals = []
        start = []
        for n, line in enumerate(readlist):
            if n==0:
                line = line.split(' ')
                line = list(map(float,line[2:]))  #handle nan values and \n in last entry
                del line[1::3]
                line = np.nan_to_num(np.array(line[:-4]))
                line0 = line
                start = line
                continue
            if n%2==0:
              line = line.split(' ')
              line = list(map(float,line[2:]))  #handle nan values and \n in last entry
              del line[1::3]
              line = np.nan_to_num(np.array(line[:-4]))
              res = list(line-line0)
              vals.append(res)
              line0=line
        if len(vals)>self.time:
          diff = len(vals)-self.time                           #just commented
          vals = vals[int(diff/2):-(diff-int(diff/2))]     #just commented       
        if len(vals)<self.time:
            diff = self.time-len(vals)
            for i in range(diff):
                if i%2==0:
                    vals.insert(0,vals[0])
                else:
                    vals.append(vals[-1])
        y = T.Tensor(vals)
        return y, start
    
class MocapDatasetLand(Dataset):
    #This data set is for landmark prediction
    def __init__(self, head_dir: list, aud_dir: list, landM_dir: list, emo_vec: list):
        #self.df = df
        self.head = head_dir
        self.aud = aud_dir
        self.landM = landM_dir
        self.emo = emo_vec
        self.time = 0
        self._indices = None
        self.transform = None

    def __len__(self):
        return len(self.head)
    
    def len(self):
        return len(self.head)
    
    def get(self, idx):
        inp = self.aud[idx]
        out = self.landM[idx]
        X, edge_index = self.readAudInp(inp)
        y = self.readLandMark(out)
        data = Data(x=X, edge_index = edge_index, y=y)
        return data, out

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
        edge_index = []
        for i in range(self.time):
            if i<self.time-1:
                edge_index.append([i,i+1])
                continue
            #if i==self.time-1:
            #    edge_index.append([i,0])
        #edge_index = np.array(edge_index,dtype=np.int64)
        edge_index = T.transpose(T.Tensor(edge_index),0,1)
        return X, edge_index.type(dtype=T.int64)

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
        #res = [y - x for x,y in pairwise(vals)]
        #print(len(vals),self.time)
        if len(vals)>self.time:
          diff = len(vals)-self.time                           #just commented
          vals = vals[int(diff/2):-(diff-int(diff/2))]     #just commented       
        y = T.Tensor(vals)
        return y