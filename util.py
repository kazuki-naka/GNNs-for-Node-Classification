# -*- coding:utf-8 -*-
import copy

import numpy as np
import torch
from tqdm import tqdm

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.datasets import NELL, Planetoid, ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[data.val_mask], data.y[data.val_mask])
    _, pred = out.max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    model.train()

    return loss.item(), acc


def train(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    min_val_loss = np.Inf
    best_model = None
    min_epochs = 5

    # Using torch profiler for how much CPU memory has been used
    with torch.profiler.profile(profile_memory=True, with_flops=True) as p: 
        model.train()
        final_test_acc = 0
        for epoch in tqdm(range(200)):
            out = model(data)
            optimizer.zero_grad()
            loss = loss_function(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # validation
            val_loss, test_acc = test(model, data)
            if val_loss < min_val_loss and epoch + 1 > min_epochs:
                min_val_loss = val_loss
                final_test_acc = test_acc
                best_model = copy.deepcopy(model)
            tqdm.write('Epoch {:03d} train_loss {:.4f} val_loss {:.4f} test_acc {:.4f}'
                    .format(epoch, loss.item(), val_loss, test_acc))
    
    with open('result.txt', 'a') as text: 
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)

    return best_model, final_test_acc

# def load_data(path, name):
#     if name == 'NELL':
#         dataset = NELL(root=path + '/NELL')
#     else:
#         dataset = Planetoid(root=path, name=name)

#     data = dataset[0].to(device)
#     if name == 'NELL':
#         data.x = data.x.to_dense()

#     return data, dataset.num_node_features, dataset.num_classes

def load_data():
    if name == 'NELL':
        dataset = NELL(root=path + '/NELL')
    else:
        # dataset = Planetoid(root=path, name=name)

        # Create Synthetic data of PubMed
        dataset = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=19717, num_edges=88648),
            motif_generator='house',
            num_motifs=80,
        )

    data = dataset[0].to(device)
    if name == 'NELL':
        data.x = data.x.to_dense()

    return data, dataset.num_node_features, dataset.num_classes

# Make distribution shift of dataset and graph
def plot_sensitivity(name, file, betas, n_moments): 
    """
    Plot sensitivity analysis
    """
    base_mmatch = 4
    base_mmd = accs_mmd.mean(1).mean(0).argmax()
    
    f, ax = plt.subplots(1, 1)
    for i in range(12):
        ax.plot(n_moments,
                file.mean(1)[i,:]/file.mean(1)[i,base_mmatch])
    ax.plot(n_moments,
            file.mean(1).mean(0)/file.mean(1).mean(0)[base_mmatch],
            'k--',linewidth=4)
    ax.grid(True,linestyle='-',color='0.75')
    plt.sca(ax)
    plt.xticks(range(1,len(n_moments)+1),n_moments)
    plt.xlabel('number of moments', fontsize=15)
    plt.ylabel('accuracy improvement', fontsize=15)
    ax.set_ylim([0.4,1.0])
    plt.savefig(name+'.png')