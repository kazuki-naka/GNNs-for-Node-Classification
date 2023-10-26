import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import time

from get_data import load_data
from models import GAT
from util import train, count_parameters, device

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"


def main():
    # name: CiteSeer Cora NELL PubMed
    t_total = time.time()
    dataset, num_in_feats, num_out_feats = load_data(path, name='PubMed')
    model = GAT(num_in_feats, 64, num_out_feats).to(device)
    model, test_acc = train(model, dataset)
    print('test acc:', test_acc)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Save model
    torch.save(model.state_dict(), 'weight_base.pth')
    count_parameters(model)

if __name__ == '__main__':
    main()