import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import time
import loralib as lora
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
    
    lora.mark_only_lora_as_trainable(model)
    update_param_names = ["conv1.lin_src.lora_A", "conv1.lin_src.lora_B", "conv1.lin_dst.lora_A", "conv1.lin_dst.lora_B",
                          "conv2.lin_src.lora_A", "conv2.lin_src.lora_B", "conv2.lin_dst.lora_A", "conv2.lin_dst.lora_B"]
    params_to_update = []
    for name, param in model.named_parameters():
        print(f"name : {name}")
        print(f"parameter : {param}")
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    # optimizer = torch.optim.Adam(params=params_to_update, lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    # print(model)

    # mem1 = psutil.virtual_memory()
    model, test_acc = train(model, dataset)
    # mem2 = psutil.virtual_memory()
    print('test acc:', test_acc)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    count_parameters(model)
    # with open('result.txt', 'a') as text: 
    #     print(f"fine-tuning used memories : {mem2.used - mem1.used}", file=text)

if __name__ == '__main__':
    main()