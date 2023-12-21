# -*- coding:utf-8 -*-
import os
import copy
import sys
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle as pkl

import networkx as nx
import scipy.sparse as sp

import matplotlib.pyplot as plt

from torch_geometric.datasets import FakeDataset, Planetoid

path = os.getcwd() + "/data"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = "Cora"

@torch.no_grad()
def test(model, data, idx_test):
    model.eval()
    out = model(data)
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(out[data.val_mask], data.y[data.val_mask])
    _, pred = out.max(dim=1)
    correct = int(pred[idx_test].eq(data.y[idx_test]).sum().item())
    acc = correct / int(idx_test.sum())
    model.train()

    return loss.item(), acc


def train(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss()
    min_val_loss = np.Inf
    best_model = None
    min_epochs = 5

    # Using torch profiler for how much CPU memory has been used
    with torch.profiler.profile(profile_memory=True, with_flops=True) as p: 
        model.train()
        for epoch in tqdm(range(200)):
            out = model(data)
            optimizer.zero_grad()
            loss = loss_function(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()  
            val_loss, test_acc = test(model, data, data.test_mask)
            if val_loss < min_val_loss and epoch + 1 > min_epochs:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)
            tqdm.write('Epoch {:03d} train_loss {:.4f} val_loss {:.4f} test_acc {:.4f}'
                    .format(epoch, loss.item(), val_loss, test_acc))
    with open('new_result.txt', 'a') as text: 
        print("train memory : ", file=text)
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)

    return best_model, test_acc

def load_data(path, name):
    dataset = Planetoid(root=path, name=name)
    data = dataset[0].to(device)
    
    return data, dataset.num_node_features, dataset.num_classes

def cmd(X, X_test, K): 
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1,mx2)
    scms = [dm]
    for i in range(K-1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1,sx2,i+2))
        #scms+=moment_diff(sx1,sx2,1)
    return sum(scms)

def l2diff(x1, x2): 
    return (x1-x2).norm(p=2)

def moment_diff(sx1, sx2, k): 
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    return l2diff(ss1,ss2)

def pairwise_distances(x, y=None): 
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def preprocess_features(features): 
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    try:
        return features.todense()
    except:
        return features

def train_test_split(data, train_ratio: float = 0.70): 
    data_len = len(data.x)
    split_num = int(data_len * train_ratio)
    train_index = random.sample(range(data_len), k = split_num)

    train_mask = torch.BoolTensor(data_len).fill_(False)

    for i in train_index: 
        train_mask[i] = True
    
    test_mask = ~train_mask

    return train_mask, test_mask