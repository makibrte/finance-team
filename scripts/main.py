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
parser = argparse.ArgumentParser(description='IMG DQN hogwild')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--save', type = str, help='Name of the model .pth file')


if __name__ == '__main__':
    args = parser.parse_args()
    
    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    data = pd.read_csv('/home/matejam/Documents/IMG/IMG/backtesting/data/AMD.csv')
    
    
    kwargs = {'batch_size':args.batch_size,
    'shuffle':False}
    if use_cuda:
        kwargs.update({
            'num_workers':1,
            'pin_memory':True,
        })
    
    torch.manual_seed(args.seed)
    mp.set_start_method('spawn', force=True)
    
    model = QNet(15, 128,3, args.save).to(device)
    agent = Agent(10000)
    model.share_memory()
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, agent, device,
        data, kwargs))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


