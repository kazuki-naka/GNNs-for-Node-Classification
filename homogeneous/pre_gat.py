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
from util import load_data, load_synthetic_data, KMM, preprocess_features, device, DATASET, ft_size, train

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"

def main():
    dataset, num_in_feats, num_out_feats = load_data(path, name=DATASET)
    model = GAT(num_in_feats, 64, num_out_feats).to(device)
    model, test_acc = train(model, dataset)
    # save model
    torch.save(model.state_dict(), 'weight_base.pth')
    with open("new_result.txt", "a") as text: 
        print('test acc:', test_acc, file = text)

if __name__ == '__main__':
    main()