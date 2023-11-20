# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from layers import GATConv
from util import cmd

# class GAT(torch.nn.Module):
#     def __init__(self, in_feats, h_feats, out_feats, finetune=False):
#         super(GAT, self).__init__()
#         self.conv1 = GATConv(in_feats, h_feats, heads=8, concat=False, finetune = finetune)
#         self.conv2 = GATConv(h_feats, out_feats, heads=8, concat=False, finetune = finetune)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)

#         return x


class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 num_heads=8, 
                 finetune=False):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(GATConv(in_feats, n_hidden, num_heads=num_heads, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GATConv(n_hidden*num_heads, n_hidden, num_heads=num_heads, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(GATConv(n_hidden*num_heads, n_classes, num_heads=1, feat_drop=dropout, activation=None, finetune=finetune)) # activation None


        #embed()
    def forward(self, features, bns=False):
        h = features
        for idx in range(len(self.layers)-1):
            h = self.layers[idx](self.g, h).flatten(1)
        self.h = h
        return self.layers[-1](self.g, h).mean(1)

    def output(self, g, features):
        h = features
        for idx in range(len(self.layers)-1):
            h = self.layers[idx](g, h).flatten(1)
        return self.layers[-1](g, h).mean(1)
    
    def shift_robust_output(self, idx_train, iid_train, K, alpha=1): 
        return alpha * cmd(self.h[idx_train, :], self.h[iid_train, :], K)