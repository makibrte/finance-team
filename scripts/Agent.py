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
from Model import QNet
import random 
class Agent:
    def __init__(self, cash, args):
        torch.set_default_device(args.device)
        self.cash = torch.tensor(cash).to(args.device)
        self.init_balance = torch.tensor(cash).to(args.device)
        self.holdings = torch.zeros(20,1).to(args.device)
        self.value = torch.tensor(cash).to(args.device)
        self.args = args
        self.memory = deque(maxlen=46800)
        self.holdings_average = torch.tensor(0).to(args.device)
        self.performance = torch.tensor(0).to(args.device)
        self.model = QNet(155,10000,3, args.save_file, args).to(args.device)
        self.model.share_memory()
        self.update_tensor = torch.tensor([[1, -1, 0] for _ in range(20)], dtype=torch.float32).to(args.device)

        
        self.epochs = torch.tensor(0)

    def getState(self, data):
        
        state = torch.cat((data, self.value.unsqueeze(0))).to(self.args.device)
        
        
        return state
    
    def getAction(self,data):
        
        self.epsilon = 100 - self.epochs
        
        
        if random.randint(0, 80) < self.epsilon:
            prediction_tensor = torch.randn(20, 3)
            max_indices = torch.argmax(prediction_tensor, dim=1)
            final_action = torch.zeros_like(prediction_tensor)
            final_action.scatter_(1, max_indices.unsqueeze(1), 1)
        else:
            #print('Model')
            state0 = data.clone()
            prediction = self.model(state0)
            max_indices = torch.argmax(prediction, dim=1)
            final_action = torch.zeros_like(prediction_tensor)
            final_action.scatter_(1, max_indices.unsqueeze(1), 1)
        return final_action
    
    def updateValue(self, action, prices) -> None:
        temp_holdings = self.holdings.clone().to(self.args.device)
        temp_holdings.to(self.args.device)
        temp_holdings = temp_holdings.to(self.args.device) + torch.sum((action.to(self.args.device) * self.update_tensor.to(self.args.device) * temp_holdings.to(self.args.device)), dim=1, keepdim=True).to(self.args.device)
        
        if torch.sum(temp_holdings < 0) > 0:
            #TODO : Change so it has different recorded value to put into the csv file
            pass
        else:
            cash_change = torch.tensor(torch.sum(prices.to(self.args.device)* torch.sum((action.to(self.args.device) * self.update_tensor.to(self.args.device) * self.holdings.to(self.args.device)), dim=1, keepdim=True))).to(self.args.device)
            if cash_change <= self.cash:
                self.holdings = self.holdings.to(self.args.device) + torch.sum((action.to(self.args.device) * self.update_tensor.to(self.args.device) * self.holdings.to(self.args.device)), dim=1, keepdim=True).to(self.args.device)
                self.cash = self.cash.to(self.args.device) - cash_change.to(self.args.device)
        self.value = torch.sum(self.holdings * prices).to(self.args.device) + self.cash.to(self.args.device)
        self.stepPerformance()
    def stepPerformance(self):
        self.performance = self.value.to(self.args.device) / self.init_balance.to(self.args.device) - self.performance.to(self.args.device)

    def finalPerformance(self):
        return self.value.to(self.args.device) / self.init_balance.to(self.args.device)
    def remember(self, state, action, reward, next_state):
        self.memory.append((action, state, reward, next_state))
