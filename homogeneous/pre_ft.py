import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir))

import torch
from tqdm import tqdm
import loralib as lora

from models import GAT
from util import device, DATASET, test
import pre_gat


def main():
    dataset = pre_gat.dataset
    num_in_feats = pre_gat.num_in_feats
    num_out_feats = pre_gat.num_out_feats
    test_mask = pre_gat.test_mask

    # divide test_mask into new train_mask and test_mask for fine-tuning
    test_len = len(test_mask)
    test_index = torch.where(test_mask == True)[0]
    split_index = torch.chunk(test_index, 2)
    ft_train_index, ft_test_index = split_index[0], split_index[1]
    ft_train_mask, ft_test_mask = torch.BoolTensor(test_len).fill_(False), torch.BoolTensor(test_len).fill_(False)
    for i in range(len(ft_train_index)): 
        ft_train_mask[ft_train_index[i].item()] = True
    
    for i in range(len(ft_test_index)): 
        ft_test_mask[ft_test_index[i].item()] = True
    
    model = GAT(num_in_feats, 64, num_out_feats, finetune = True, r1 = 1, r2 = 1).to(device)

    # load pre-trained model
    model.load_state_dict(torch.load('weight_base.pth'), strict=False)
    lora.mark_only_lora_as_trainable(model)

    update_param_names = ["conv1.lin.lora_A", "conv1.lin.lora_B", "conv2.lin.lora_A", "conv2.lin.lora_B"]
    # update_param_names = ["conv1.lin.lora_A", "conv1.lin.lora_B"]
    # update_param_names = ["conv2.lin.lora_A", "conv2.lin.lora_B"]
    params_to_update = []
    for name, param in model.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    
    with open("new_result.txt", "a") as text: 
        print('fine-tuning', file=text)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    # Using torch profiler for how much CPU memory has been used
    with torch.profiler.profile(profile_memory=True, with_flops=True) as p: 
        model.train()
        for epoch in tqdm(range(200)):
            out = model(dataset)
            optimizer.zero_grad()
            loss = loss_function(out[ft_train_mask], dataset.y[ft_train_mask])
            loss.backward()
            optimizer.step()
    with open('new_result.txt', 'a') as text: 
        print("train(fine-tuning) : ", file=text)
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)
        print("test(fine-tuning) : ", file=text)
    

    with torch.profiler.profile(profile_memory=True, with_flops=True) as p:
        val_loss, test_acc = test(model, dataset, ft_test_mask)
    with open('new_result.txt', 'a') as text: 
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10), file=text)
        print("test acc: ", test_acc,  file=text)
        print("\n", file=text)
        
if __name__ == '__main__':
    main()