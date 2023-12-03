import os
import copy
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import torch.nn  as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import networkx as nx
import scipy.sparse as sp
import numpy as np
import dgl
import pickle as pkl

from models import GAT
from util import load_data, load_synthetic_data, KMM, preprocess_features, device, DATASET, train, test, cmd, path


def main():
    dataset, num_in_feats, num_out_feats = load_data(path, name=DATASET)
    model = GAT(num_in_feats, 64, num_out_feats).to(device)
    with open("new_result.txt", "w") as text: 
        print('pre-train', file = text)
    dataset = dataset.to(device)
    model, test_acc = train(model, dataset)
    # save model
    torch.save(model.state_dict(), 'weight_base.pth')

    idx_train = torch.LongTensor(pkl.load(open('{}/{}/raw/localized_seeds_{}.p'.format(path, DATASET, DATASET.lower()), 'rb'))[0])
    all_idx = set(range(dataset.num_nodes)) - set(idx_train)
    idx_test = torch.LongTensor(list(all_idx))

    model.conv2 = nn.Identity()
    feature = model(dataset)
    X = feature[idx_train, :]
    X_test = feature[idx_test, :]
    value_cmd = cmd(X, X_test, K=1)

    with torch.profiler.profile(profile_memory=True, with_flops=True) as p: 
        val_loss, test_acc = test(model, dataset, idx_test)
    with open('new_result.txt', 'a') as text: 
        print("test memory : ", file=text)
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)
        print("test acc: ", test_acc, file=text)
        print("value of cmd: ", value_cmd, file=text)
        print("\n", file=text)

if __name__ == '__main__':
    main()