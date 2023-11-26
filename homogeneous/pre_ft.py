import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import torch.nn  as nn
import torch.nn.functional as F
import time
import loralib as lora
import torch.nn.utils.prune as prune

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import dgl
import pickle as pkl
from sklearn.metrics import f1_score
import scipy.sparse as sp
from models import GAT
from util import load_data, load_synthetic_data, KMM, cmd, preprocess_features, device, DATASET, ft_size, train

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"


def main():
    dataset, num_in_feats, num_out_feats = load_data(path, name=DATASET)
    model = GAT(num_in_feats, 64, num_out_feats, finetune = True).to(device)
    # load pre-trained model
    model.load_state_dict(torch.load('weight_base.pth'), strict=False)
    lora.mark_only_lora_as_trainable(model)

    idx_train = dataset.train_mask
    perm = torch.randperm(data.test_mask.shape[0])

    # independent and identically distributed sampling
    iid_train = dataset.test_mask[perm[:idx_train.shape[0]]]
    dataset.test_mask = iid_train

    update_param_names = ["conv1.lin_src.lora_A", "conv1.lin_src.lora_B"]
    params_to_update = []
    for name, param in model.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    model, test_acc = train(model, dataset)
    
    X = self.h_out([idx_train, :])
    X_test = self.h_out([iid_train, :])
    value_cmd = cmd(X, X_test, K=5)

    with open("new_result.txt", "a") as text: 
        print('test acc:', test_acc, file=text)
        print('cmd value: ', value_cmd, file=text)
if __name__ == '__main__':
    main()