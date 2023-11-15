import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import time

from models import GAT
from util import load_data, train, device

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"

def main():
    dataset, num_in_feats, num_out_feats = load_data(path, name='Cora')
    model = GAT(num_in_feats, 64, num_out_feats).to(device)
    t_total = time.time()
    model, test_acc = train(model, dataset)
    print('test acc:', test_acc)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Save model
    torch.save(model.state_dict(), 'weight_base.pth')

if __name__ == '__main__':
    main()