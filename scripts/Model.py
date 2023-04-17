from __future__ import print_function
import argparse

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms 
from collections import deque
from train import train, test
import random
class QNet(nn.Module):
    def __init__(self, input, hidden, output, save_file, args):
        super(QNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden).to(args.device)
        self.linear2 = nn.Linear(hidden, output).to(args.device)
        self.save_file = save_file
        self.args = args
        weight_dtype = self.linear1.weight.dtype
        print("Data type required by the model:", weight_dtype)

    
    def forward(self, x):
        x = F.relu(self.linear1(x)).to(self.args.device)
        x = F.dropout(x, training=self.training).to(self.args.device)
        x = self.linear2(x).to(self.args.device)

        return x.view(-1,20,3)
    def save(self):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, self.save_file)
        torch.save(self.state_dict(), file_name)