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
    model, test_acc = train(model, dataset)

    # distribution shift on graphs
    node_feats, y = load_synthetic_data(path, DATASET,1)
    dataset.x = node_feats
    dataset.y = y
    with torch.profiler.profile(profile_memory=True, with_flops=True) as p: 
        val_loss, test_acc = test(model, dataset)
    with open('new_result.txt', 'a') as text: 
        print("test memory : ", file=text)
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)
    
    # save model
    torch.save(model.state_dict(), 'weight_base.pth')
    with open("new_result.txt", "a") as text: 
        print("test acc: ", test_acc, file=text)
        print("\n", file=text)

if __name__ == '__main__':
    main()