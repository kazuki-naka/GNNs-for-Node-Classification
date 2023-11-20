import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import torch.nn  as nn
import torch.nn.functional as F
import time
import networkx as nx
import scipy.sparse as sp
import numpy as np
import dgl
import pickle as pkl
from sklearn.metrics import f1_score

from models import GAT
from util import load_data, load_synthetic_data, KMM, preprocess_features, device, DATASET, ft_size

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"

def main():
    # dataset, num_in_feats, num_out_feats = load_data(path, name='Cora')
    # dataset, num_in_feats, num_out_feats = load_synthetic_data()
    adj, features, one_hot_labels, ori_idx_train, idx_val, idx_test = load_data(DATASET)
    nx_g = nx.Graph(adj+ sp.eye(adj.shape[0]))
    g = dgl.from_networkx(nx_g).to(device)
    labels = torch.LongTensor([np.where(r==1)[0][0] if r.sum() > 0 else -1 for r in one_hot_labels]).to(device)
    features = torch.FloatTensor(preprocess_features(features)).to(device)
    xent = nn.CrossEntropyLoss().to(device)

    nb_classes = max(labels).item() + 1

    # model = GAT(num_in_feats, 64, num_out_feats).to(device)
    # The point to be revised
    model = GAT(g, ft_size, 32, nb_classes, 2, F.tanh, 0.5)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    model.cuda()

    idx_train = torch.LongTensor(pkl.load(open('data/{}/raw/localized_seeds_{}.p'.format(DATASET, DATASET.lower()), 'rb'))[0])
    all_idx = set(range(g.number_of_nodes())) - set(idx_train)
    idx_test = torch.LongTensor(list(all_idx))
    perm = torch.randperm(idx_test.shape[0])
    iid_train = idx_test[perm[:idx_train.shape[0]]]

    # Z_train = torch.FloatTensor(adj[idx_train.tolist(), :].todense())
    # Z_test = torch.FloatTensor(adj[iid_train.tolist(), :].todense())
    # label_balance_constraints = np.zeros((labels.max().item()+1, len(idx_train)))
    # for i, idx in enumerate(idx_train):
    #     label_balance_constraints[labels[idx], i] = 1
    # kmm_weight = KMM(Z_train, Z_test, label_balance_constraints, beta=0.2)

    t_total = time.time()
    with torch.profiler.profile(profile_memory=True, with_flops=True) as p:
        for epoch in range(200):
            model.train()
            optimiser.zero_grad()
            logits = model(features)
            loss = xent(logits[idx_train], labels[idx_train])
            # loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean() + model.shift_robust_output(idx_train, iid_train)
            loss.backward()
            optimiser.step()
    with open('new_result.txt', 'a') as text: 
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)

    model.eval()
    embeds = model(features).detach()
    logits = embeds[idx_test]
    preds_all = torch.argmax(embeds, dim=1)

    with open('new_result.txt', 'a') as text: 
        print("Accuracy:{}".format(f1_score(labels[idx_test].cpu(), preds_all[idx_test].cpu(), average='micro')), file=text)
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total), file=text)

if __name__ == '__main__':
    main()