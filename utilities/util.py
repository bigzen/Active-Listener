
import csv
import torch as T
import os
import random
import numpy as np
from torch.nn.functional import pad

def TrTstSplit(dir_list):
    tr_splt = len(dir_list)*4/5
    random.shuffle(dir_list)
    train = dir_list[0:int(tr_splt)]
    test = dir_list[int(tr_splt):]
    print(train)
    return train, test

def colate_fn(data):
    x_m = 0
    for x, _,_,_,_ in data:
        if x.shape[-2]>x_m:
            x_m = x.shape[-2]
    X_n = []
    land_n = []
    hea_n = []
    em = []
    val = []
    for x, l, h, e, v in data:
        X_n.append(pad(x,(0,0,0,x_m-x.shape[-2])))
        land_n.append(pad(l,(0,0,0,x_m-x.shape[-2])))
        #print(x.shape,X_n[-1].shape, l.shape)
        hea_n.append(pad(h,(0,0,0,x_m-x.shape[-2])))
        em.append(e)
        val.append(v)
    X = T.stack(X_n, dim=0)
    #for l in land_n:
    #    print(l.shape)
    land = T.stack(land_n, dim=0)
    hea = T.stack(hea_n, dim=0)
    val = T.stack(val, dim=0)
    return [X, land, hea, em, T.unsqueeze(val, 1)]

def colate_fn2(data):
    x_m = 0
    for x, _,_,_,_ in data:
        if x.shape[-2]>x_m:
            x_m = x.shape[-2]
    X_n = []
    land_n = []
    hea_n = []
    em = []
    val = []
    shape = []
    for x, l, h, e, v in data:
        xl = x.shape[-2]
        x_fill = x[:,-1,:]
        x2 = pad(x,(0,0,0,x_m-x.shape[-2]))
        x2[:,xl:,:]=x_fill        
        X_n.append(x2)
        shape.append(l.shape[1])
        l_fill = l[:,-1,:]
        l2 = pad(l,(0,0,0,x_m-x.shape[-2]))
        l2[:,xl:,:]=l_fill
        land_n.append(l2)
        h_fill = h[:,-1,:]
        h2 = pad(h,(0,0,0,x_m-x.shape[-2]))
        h2[:,xl:,:]=h_fill
        hea_n.append(h2)
        em.append(e)
        val.append(v)
    X = T.stack(X_n, dim=0)
    land = T.stack(land_n, dim=0)
    hea = T.stack(hea_n, dim=0)
    val = T.stack(val, dim=0)
    return [X, land, hea, shape, em, T.unsqueeze(val, 1)]
    
def GetInputOutputSplit(dir_list):
    head = []
    aud = []
    landM = []
    emo = []
    val = []
    for dirs in dir_list:
        with open(dirs,'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            readlist = np.array(list(csv_reader))
            head.extend(list(readlist[:,0]))
            aud.extend(list(readlist[:,1]))
            landM.extend(list(readlist[:,2]))
            emo.extend(list(readlist[:,3]))
            val.extend(list(readlist[:,4]))
    return head, aud, landM, emo, val

def FaceGraph():
    dire = '../backchannel_gesture/utilities/land.csv'
    graph = []
    with open(dire,'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        readlist = list(csv_reader)[1:]
        for line in readlist:
            graph.append(line[1:])
        grp = np.nan_to_num(np.array(graph))
        edge = np.argwhere(grp=='1')
    edge = np.transpose(edge)
    return edge

