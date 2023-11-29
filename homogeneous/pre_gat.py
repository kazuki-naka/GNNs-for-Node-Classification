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
    test_acc = train(model, data)
    # save model
    torch.save(model.state_dict(), 'weight_base.pth')
    with open("new_result.txt", "a") as text: 
        print("test acc: ", test_acc, file=text)
        print("\n", file=text)

if __name__ == '__main__':
    main()