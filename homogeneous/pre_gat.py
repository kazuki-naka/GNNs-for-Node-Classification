import os
import copy
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp
import numpy as np
import random

from models import GAT
from util import load_data, load_synthetic_data, KMM, preprocess_features, device, DATASET, train, test, cmd, path, train_test_split


def main():
    dataset, num_in_feats, num_out_feats = load_data(path, name=DATASET)
    model = GAT(num_in_feats, 64, num_out_feats).to(device)
    with open("new_result.txt", "w") as text: 
        print('pre-train', file = text)
    dataset = dataset.to(device)
    model, test_real_acc = train(model, dataset)
    # save model
    torch.save(model.state_dict(), 'weight_base.pth')

    train_mask, test_mask = train_test_split(dataset)
    Z = model.feature[dataset.train_mask, :]
    Z_l = model.feature[test_mask, :]
    value_cmd = cmd(Z, Z_l, K=5).item()

    with torch.profiler.profile(profile_memory=True, with_flops=True) as p: 
        val_loss, test_acc = test(model, dataset, test_mask)
        result = np.array([value_cmd, test_acc])
        np.save('{}/data_5'.format(os.path.abspath(os.path.dirname(__file__))), result)
    with open('new_result.txt', 'a') as text: 
        print("test memory : ", file=text)
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)
        print("test acc(real data): ", test_real_acc, file=text)
        print("test acc: ", test_acc, file=text)
        print("value of cmd: ", value_cmd, file=text)
        print("\n", file=text)

if __name__ == '__main__':
    main()