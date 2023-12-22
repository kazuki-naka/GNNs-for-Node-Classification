import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import numpy as np

from models import GAT
from util import load_data, device, DATASET, train, test, cmd, path, train_test_split


dataset, num_in_feats, num_out_feats = load_data(path, name=DATASET)
train_mask, test_mask = train_test_split(dataset)

def main():
    global dataset, num_in_feats, num_out_feats, train_mask, test_mask
    dataset.train_mask = train_mask
    model = GAT(num_in_feats, 64, num_out_feats).to(device)
    with open("new_result.txt", "a") as text: 
        print('pre-train', file = text)
    model, test_real_acc = train(model, dataset)
    # save model
    torch.save(model.state_dict(), 'weight_base.pth')

    Z = model.feature[dataset.train_mask, :]
    Z_l = model.feature[test_mask, :]
    value_cmd = cmd(Z, Z_l, K=5).item()
    val_loss, test_acc = test(model, dataset, test_mask)
    with open('new_result.txt', 'a') as text: 
        print("test acc(real data): ", test_real_acc, file=text)
        print("test acc: ", test_acc, file=text)
        print("value of cmd: ", value_cmd, file=text)
        print("\n", file=text)

if __name__ == '__main__':
    main()