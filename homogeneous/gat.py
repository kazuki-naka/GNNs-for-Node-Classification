import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import time
import psutil

from get_data import load_data
from models import GAT
from util import train, count_parameters, device

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"


def main():
    # name: CiteSeer Cora NELL PubMed
    t_total = time.time()
    dataset, num_in_feats, num_out_feats = load_data(path, name='PubMed')
    model = GAT(num_in_feats, 64, num_out_feats).to(device)
    mem1 = psutil.virtual_memory()
    model, test_acc = train(model, dataset)
    mem2 = psutil.virtual_memory()
    print('test acc:', test_acc)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Save model
    torch.save(model.state_dict(), 'weight_base.pth')
    count_parameters(model)
    with open('result.txt', 'w') as text: 
        print(f"pre_train used memories : {mem2.used - mem1.used}", file=text)

if __name__ == '__main__':
    main()