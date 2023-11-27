# -*- coding:utf-8 -*-
import os
import copy
import sys

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = "Cora"

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[data.val_mask], data.y[data.val_mask])
    _, pred = out.max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    model.train()

    return loss.item(), acc


# def train(model, data, idx_train, iid_train):
#     plot_x, plot_y = [],[]
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
#     loss_function = torch.nn.CrossEntropyLoss().to(device)
#     min_val_loss = np.Inf
#     best_model = None
#     min_epochs = 5
#     # Using torch profiler for how much CPU memory has been used
#     with torch.profiler.profile(profile_memory=True, with_flops=True) as p: 
#         model.train()
#         final_test_acc = 0
#         for epoch in tqdm(range(200)):
#             out = model(data)
#             optimizer.zero_grad()
#             loss = loss_function(out[idx_train], data.y[iid_train])
#             loss.backward()
#             optimizer.step()

#             # Test
#             val_loss, test_acc = test(model, data)
#             if val_loss < min_val_loss and epoch + 1 > min_epochs:
#                 min_val_loss = val_loss
#                 final_test_acc = test_acc
#                 best_model = copy.deepcopy(model)
#             tqdm.write('Epoch {:03d} train_loss {:.4f} val_loss {:.4f} test_acc {:.4f}'
#                     .format(epoch, loss.item(), val_loss, test_acc))
            
#             plot_x.append(cmd(model.h_out[idx_train, :], model.h_out[iid_train, :], K=5).item())
#             plot_y.append(test_acc)
#     with open('new_result.txt', 'a') as text: 
#         print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)
#     plt.scatter(plot_x, plot_y)
#     plt.xlim(0, 5)
#     plt.savefig("sample.png")

#     return best_model, final_test_acc

def load_data(path, name):
    dataset = Planetoid(root=path, name=name)
    data = dataset[0].to(device)
    return data, dataset.num_node_features, dataset.num_classes

# def load_data(dataset_str): 
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("data/{}/raw/ind.{}.{}".format(dataset_str, dataset_str.lower(), names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))

#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("data/{}/raw/ind.{}.test.index".format(dataset_str, dataset_str.lower()))
#     test_idx_range = np.sort(test_idx_reorder)
#     # embed()
#     if dataset_str == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range-min(test_idx_range), :] = ty
#         ty = ty_extended
#     # embed()
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#     #embed()
#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]

#     idx_test = test_idx_range.tolist()
#     #embed()
#     #idx_train = range(len(y))
#     if dataset_str.lower() == 'pubmed':
#         idx_train = range(10000)
#     elif dataset_str.lower() == 'cora':
#         idx_train = range(1500)
#     else:
#         idx_train = range(1000)
#     idx_val = range(len(y), len(y)+500)
#     return adj, features, labels, idx_train, idx_val, idx_test

def load_synthetic_data(): 
    dataset = FakeDataset(num_channels=1433, num_classes=7, task='node')
    data = dataset[0].to(device)

    return data, dataset.num_node_features, dataset.num_classes

# def parse_index_file(filename): 
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index

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

# find the appropriate weights which minimize MMD distance
def KMM(X,Xtest,_A=None, _sigma=1e1,beta=0.2):

    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(- 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    H /= 3
    f /= 3
    
    nsamples = X.shape[0]
    f = - X.shape[0] / Xtest.shape[0] * f.matmul(torch.ones((Xtest.shape[0],1)))
    G = - np.eye(nsamples)
    _A = _A[~np.all(_A==0, axis=1)]
    b = _A.sum(1)
    h = - beta * np.ones((nsamples,1))
    
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    sol=solvers.qp(matrix(H.numpy().astype(np.double)), matrix(f.numpy().astype(np.double)), matrix(G), matrix(h), matrix(_A), matrix(b))
    return np.array(sol['x'])

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
    
# def MMD(X, Xtest): 
#     H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(- 1e-3 * pairwise_distances(X))
#     f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(X, Xtest))
#     z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
#     MMD_dist = H.mean() - 2 * f.mean() + z.mean()
#     return MMD_dist