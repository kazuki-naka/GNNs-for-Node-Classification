# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from layers import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, finetune=False):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=8, concat=False, finetune = finetune)
        self.conv2 = GATConv(h_feats, out_feats, heads=8, concat=False, finetune = finetune)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x
