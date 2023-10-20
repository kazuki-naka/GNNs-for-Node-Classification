import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import time
import psutil
import torch.nn.utils.prune as prune

from get_data import load_data
from models import GAT
from util import train, count_parameters, device

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"


def main():
    # name: CiteSeer Cora NELL PubMed
    t_total = time.time()
    dataset, num_in_feats, num_out_feats = load_data(path, name='PubMed')
    model = GAT(num_in_feats, 64, num_out_feats, finetune = True).to(device)
    # load pre-trained model
    model.load_state_dict(torch.load('weight_base.pth'), strict=False)

    # Fine-tuning, freeze layers' weight except last layer
    for layer in model.parameters(): 
        layer.requires_grad = False
    
    for param in model.conv1.lin_dst.parameters(): 
        param.requires_grad = True

    for param in model.conv2.lin_dst.parameters(): 
        param.requires_grad = True
    
    model, test_acc = train(model, dataset)
    print('test acc:', test_acc)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    count_parameters(model)
    with open('result.txt', 'a') as text: 
        mem = psutil.virtual_memory()
        print(f"fine-tuning used memories : {mem.used}", file=text)

if __name__ == '__main__':
    main()