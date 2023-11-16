import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import time
import loralib as lora
import torch.nn.utils.prune as prune

import networkx as nx
import scipy.sparse as sp
from models import GAT
from util import train, load_data, load_synthetic_data, KMM,preprocess_features, device, DATASET

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"


def main():
    # dataset, num_in_feats, num_out_feats = load_data(path, name='Cora')
    # dataset, num_in_feats, num_out_feats = load_synthetic_data()
    adj, features, one_hot_labels, ori_idx_train, idx_val, idx_test = load_data(DATASET)
    nx_g = nx.Graph(adj+ sp.eye(adj.shape[0]))
    g = dgl.from_networkx(nx_g).to(device)
    labels = torch.LongTensor([np.where(r==1)[0][0] if r.sum() > 0 else -1 for r in one_hot_labels]).to(device)
    features = torch.FloatTensor(preprocess_features(features)).to(device)
    xent = nn.CrossEntropyLoss(reduction='none')

    idx_train = torch.LongTensor(pickle.load(open('data/{0}/raw/localized_seeds_{1}.p'.format(DATASET, DATASET), 'rb'))[0])
    all_idx = set(range(g.number_of_nodes())) - set(idx_train)
    idx_test = torch.LongTensor(list(all_idx))
    perm = torch.randperm(idx_test.shape[0])
    iid_train = idx_test[perm[:idx_train.shape[0]]]

    Z_train = torch.FloatTensor(adj[idx_train.tolist(), :].todense())
    Z_test = torch.FloatTensor(adj[iid_train.tolist(), :].todense())
    label_balance_constraints = np.zeros((labels.max().item()+1, len(idx_train)))
    for i, idx in enumerate(idx_train):
        label_balance_constraints[labels[idx], i] = 1
    kmm_weight, MMD_dist = KMM(Z_train, Z_test, label_balance_constraints, beta=0.2)

    model = GAT(num_in_feats, 64, num_out_feats, finetune = True).to(device)
    # load pre-trained model
    model.load_state_dict(torch.load('weight_base.pth'), strict=False)
    lora.mark_only_lora_as_trainable(model)
    # update_param_names = ["conv1.lin_src.lora_A", "conv1.lin_src.lora_B", "conv1.lin_dst.lora_A", "conv1.lin_dst.lora_B",
    #                       "conv2.lin_src.lora_A", "conv2.lin_src.lora_B", "conv2.lin_dst.lora_A", "conv2.lin_dst.lora_B"]
    update_param_names = ["conv1.lin_src.lora_A", "conv1.lin_src.lora_B"]
    params_to_update = []
    for name, param in model.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    t_total = time.time()
    model, test_acc = train(model, dataset)
    print('test acc:', test_acc)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if __name__ == '__main__':
    main()