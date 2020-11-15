import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np
import os
import torch
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from Data.MyDataSet import Trian_DataSet
from models.MyModel import UNet_3Plus


pretrain_model_path = "../save/unet3.pt"
epoch_save_model_path = "../save/epoch.pt"
final_save_model_path = "../save/unet3.pt"

def calc_time(start_time, end_time):
    s = int(end_time - start_time)
    m = int(s / 60)
    h = int(m / 60)
    s %= 60
    m %= 60
    return h, m, s

def train(model, device):
    model = model.train()
    lr = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    train_set = Trian_DataSet()
    batch_size = 4
    num_worker = 1
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_worker,
                              pin_memory=True, sampler=None)
    total_loss = 0
    for i, data in enumerate(train_loader):
        data = {
            'image':           data[0].to(device),
            'label':           data[1].to(device),
        }

        outputs = model.train_forword(data)
        loss = outputs['total_loss']

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if i % 200 == 0 and i != 0:
            torch.save(model.state_dict(), final_save_model_path)
            print("avg loss is {0}".format(total_loss / i))
    torch.save(model.state_dict(), final_save_model_path)
    print("epoch avg loss is {0}".format(total_loss / train_set.size))



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_3Plus()
    # 加载模型写在to(device之前)
    if os.path.exists(pretrain_model_path):
        model.load_state_dict(torch.load(pretrain_model_path))
    model = model.to(device)
    epoch = 10
    totle_epoch_start_time = time.time()
    for i in range(epoch):
        one_epoch_start_time = time.time()
        train(model, device)
        torch.save(model.state_dict(), epoch_save_model_path)
        one_epoch_end_time = time.time()
        h, m, s = calc_time(one_epoch_start_time, one_epoch_end_time)
        print("epoch{0} 花费时间{1}h{2}m{3}s".format(i, h, m, s))
    totle_epoch_end_time = time.time()
    h, m, s = calc_time(totle_epoch_start_time, totle_epoch_end_time)
    print("训练{0}轮， 共花费{1}h{2}m{3}s".format(epoch, h, m, s))


if __name__ == '__main__':
    main()


