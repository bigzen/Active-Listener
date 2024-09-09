import torch
from torch.nn import ReLU
from  torch_geometric.nn import aggr, SAGEConv, MessagePassing, GCNConv, GatedGraphConv, Linear, Sequential
from utilities.util import FaceGraph
import torch as T
import numpy as np
from scipy import signal

class GCN(torch.nn.Module):
    def __init__(self, hidden, classes):
        super(GCN, self).__init__()
        self.lin1 = Linear(int(hidden/2), int(hidden/2))
        self.lin2 = Linear(int(hidden/2), int(hidden*1.5))
        self.lin3 = Linear(int(hidden*1.5), int(hidden/2))
        self.conv1 = SAGEConv(int(hidden/2), int(hidden/2),aggr="lstm", project=True)
        self.conv2 = SAGEConv(int(hidden/2), classes, aggr="lstm", project=True)
        



    def forward(self, data):
        [x, edge_index] = data
        h = self.lin1(x)
        h = h.relu()  
        h = self.lin2(h)
        h = h.relu()  
        h = self.lin3(h)
        h = h.relu()
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)


        return h
    
class LSTM(T.nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.relu = T.nn.ReLU()
        self.tanh = T.nn.Tanh()
        self.fc1 = T.nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.bn1 = T.nn.BatchNorm1d(int(hidden_size/2))
        self.fc2 = T.nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.bn2 = T.nn.BatchNorm1d(int(hidden_size/2))
        self.lstm = T.nn.LSTM(int(hidden_size/2), num_classes, self.num_layers, batch_first=True, dropout=0.2)

    
    def forward(self, x):
        x = x.float()
        if len(x.shape)>2:
            h0 = T.autograd.Variable(T.zeros(self.num_layers, x.shape[0], self.num_classes).float()).cuda() 
            c0 = T.autograd.Variable(T.zeros(self.num_layers, x.shape[0], self.num_classes).float()).cuda()
        else:
            h0 = T.autograd.Variable(T.zeros(self.num_layers, self.num_classes).float()).cuda() 
            c0 = T.autograd.Variable(T.zeros(self.num_layers, self.num_classes).float()).cuda()
        out = T.swapaxes(self.bn1(T.swapaxes(self.relu(self.fc1(x)),1,2)),1,2)
        out = T.swapaxes(self.bn2(T.swapaxes(self.relu(self.fc2(out)),1,2)),1,2)
        out, _ = self.lstm(out, (h0,c0))
        return out