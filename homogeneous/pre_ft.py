import os
import copy
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import torch.nn  as nn
import torch.nn.functional as F
from tqdm import tqdm
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
from util import load_data, load_synthetic_data, KMM, cmd, preprocess_features, device, DATASET, test

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"


def main():
    dataset, num_in_feats, num_out_feats = load_data(path, name=DATASET)
    model = GAT(num_in_feats, 64, num_out_feats, finetune = True).to(device)

    idx_train = torch.LongTensor(pkl.load(open('data/{}/raw/localized_seeds_{}.p'.format(DATASET, DATASET.lower()), 'rb'))[0])
    all_idx = set(range(dataset.x.shape[0])) - set(idx_train)
    idx_test = torch.LongTensor(list(all_idx))
    perm = torch.randperm(idx_test.shape[0])
    iid_train = idx_test[perm[:idx_train.shape[0]]]

    label_balance_constraints = np.zeros((dataset.y.max().item()+1, len(idx_train)))
    for i, idx in enumerate(idx_train):
        label_balance_constraints[dataset.y[idx], i] = 1

    # load pre-trained model
    model.load_state_dict(torch.load('weight_base.pth'), strict=False)
    lora.mark_only_lora_as_trainable(model)

    update_param_names = ["conv1.lin.lora_A", "conv1.lin.lora_B", "conv2.lin.lora_A", "conv2.lin.lora_B"]
    params_to_update = []
    for name, param in model.named_parameters():
        # print(f"name : {name}")
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    
    with open("new_result.txt", "a") as text: 
        print('fine-tuning', file=text)
    
    plot_x, plot_y = [],[]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    # Using torch profiler for how much CPU memory has been used
    with torch.profiler.profile(profile_memory=True, with_flops=True) as p: 
        model.train()
    with open('new_result.txt', 'a') as text: 
        print("train(fine-tuning) : ", file=text)
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)
    for epoch in tqdm(range(200)):
        out = model(dataset)
        optimizer.zero_grad()
        loss = loss_function(out[idx_train], dataset.y[iid_train])
        # kmm_weight = KMM(model.h_out[idx_train, :], model.h_out[iid_train, :], label_balance_constraints, beta=0.2)
        # print((torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean())
        # print(cmd(model.h_out[idx_train, :], model.h_out[iid_train, :], K=5))
        # sys.exit(1)
        # loss = (torch.Tensor(kmm_weight).reshape(-1).cuda() * (loss)).mean() + cmd(model.h_out[idx_train, :], model.h_out[iid_train, :], K=5)
        loss.backward()
        optimizer.step()

        # Test
        # val_loss, test_acc = test(model, dataset)
        # tqdm.write('Epoch {:03d} train_loss {:.4f} val_loss {:.4f} test_acc {:.4f}'
        #         .format(epoch, loss.item(), val_loss, test_acc))
        
        # plot_x.append(cmd(model.h_out[idx_train, :], model.h_out[iid_train, :], K=5).item())
        # plot_y.append(test_acc)
    # plt.scatter(plot_x, plot_y)
    # plt.xlim(0, 5)
    # plt.savefig("sample.png")
    with torch.profiler.profile(profile_memory=True, with_flops=True) as p: 
        test_acc = test(model, dataset)
    with open("new_result.txt", "a") as text:  
        print("test(fine-tuning) : ", file=text)
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)
        print("test acc: ", test_acc,  file=text)
        print("\n", file=text)
if __name__ == '__main__':
    main()