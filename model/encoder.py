import torch
from torch.nn import ReLU
from  torch_geometric.nn import aggr, SAGEConv, MessagePassing, GCNConv, GatedGraphConv, Linear, Sequential
from utilities.util import FaceGraph
import torch as T
import numpy as np
from scipy import signal

class GCN(torch.nn.Module):
    def __init__(self, feat, hidden):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(feat,int(hidden/2),aggr="lstm", project=True)
        self.conv2 = SAGEConv(int(hidden/2), hidden,aggr="lstm", project=True)
        self.lin1 = Linear(hidden, int(hidden*1.5))
        self.lin2 = Linear(int(hidden*1.5), int(hidden/2))
        self.lin3 = Linear(int(hidden/2), int(hidden/2))


    def forward(self, data):
        [x, edge_index] = data
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.lin1(h)
        h = h.relu()  
        h = self.lin2(h)
        h = h.relu()  
        h = self.lin3(h)
        h = h.relu()  
        return [h, edge_index]
    
class LSTM(T.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.relu = T.nn.ReLU()
        self.tanh = T.nn.Tanh()
        self.lstm = T.nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = T.nn.Linear(hidden_size, int(hidden_size*1.5))
        self.bn1 = T.nn.BatchNorm1d(int(hidden_size*1.5))
        self.fc2 = T.nn.Linear(int(hidden_size*1.5), int(hidden_size/2))
        self.bn2 = T.nn.BatchNorm1d(int(hidden_size/2))

    
    def forward(self, x):
        x = x.float()
        if len(x.shape)>2:
            h0 = T.autograd.Variable(T.zeros(self.num_layers, x.shape[0], self.hidden_size).float()).cuda() 
            c0 = T.autograd.Variable(T.zeros(self.num_layers, x.shape[0], self.hidden_size).float()).cuda()
        else:
            h0 = T.autograd.Variable(T.zeros(self.num_layers, self.hidden_size).float()).cuda() 
            c0 = T.autograd.Variable(T.zeros(self.num_layers, self.hidden_size).float()).cuda()
        out, _ = self.lstm(x, (h0,c0))
        out = T.swapaxes(self.bn1(T.swapaxes(self.relu(self.fc1(out)),1,2)),1,2)
        out = T.swapaxes(self.bn2(T.swapaxes(self.relu(self.fc2(out)),1,2)),1,2)
     
        return out