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
from sklearn.metrics import f1_score

from models import GAT
from util import load_data, load_synthetic_data, KMM, preprocess_features, device, DATASET, test, cmd

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"

def main():
    dataset, num_in_feats, num_out_feats = load_data(path, name=DATASET)
    model = GAT(num_in_feats, 64, num_out_feats).to(device)
    with open("new_result.txt", "a") as text: 
        print('pre-train', file = text)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    # Using torch profiler for how much CPU memory has been used
    with torch.profiler.profile(profile_memory=True, with_flops=True) as p: 
        model.train()
        for epoch in tqdm(range(200)):
            out = model(dataset)
            optimizer.zero_grad()
            loss = loss_function(out[dataset.train_mask], dataset.y[dataset.train_mask])
            loss.backward()
            optimizer.step()

            # Test
            val_loss, test_acc = test(model, dataset)
            tqdm.write('Epoch {:03d} train_loss {:.4f} val_loss {:.4f} test_acc {:.4f}'
                    .format(epoch, loss.item(), val_loss, test_acc))
            
    with open('new_result.txt', 'a') as text: 
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)

    # save model
    torch.save(model.state_dict(), 'weight_base.pth')
    with open("new_result.txt", "a") as text: 
        print("test acc: ", test_acc, file=text)
        print("\n", file=text)

if __name__ == '__main__':
    main()