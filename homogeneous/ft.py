import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
import time
import loralib as lora
import torch.nn.utils.prune as prune

from models import GAT
from util import train, load_data, device

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"


def main():
    dataset, num_in_feats, num_out_feats = load_data(path, name='Cora')
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