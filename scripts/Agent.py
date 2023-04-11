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
        
        self.cash = torch.tensor(cash).to("cuda:0")
        self.init_balance = torch.tensor(cash).to("cuda:0")
        self.holdings = torch.zeros(20,1).to("cuda:0")
        self.value = torch.tensor(cash).to("cuda:0")
        
        self.memory = deque(maxlen=46800)
        self.holdings_average = torch.tensor(0).to("cuda:0")
        self.performance = torch.tensor(0).to("cuda:0")
        self.model = QNet(155,1000000,3, args.save_file).to("cuda:0")
        self.model.share_memory()
        self.update_tensor = torch.tensor([[1, -1, 0] for _ in range(20)], dtype=torch.float32).to("cuda:0")

        
        self.epochs = torch.tensor(0)

    def getState(self, data):
        
        state = torch.cat((data, torch.tensor([self.balance])))
        
        
        return state
    
    def getAction(self,data):
        
        self.epsilon = 100 - self.epochs
        
        
        if random.randint(0, 80) < self.epsilon:
            prediction_tensor = torch.randn(20, 3)
            max_indices = torch.argmax(prediction_tensor, dim=1)
            final_action = torch.zeros_like(prediction_tensor)
            final_action(1, max_indices.unsqueeze(1), 1)
        else:
            #print('Model')
            state0 = data.clone()
            prediction = self.model(state0)
            max_indices = torch.argmax(prediction, dim=1)
            final_action = torch.zeros_like(prediction_tensor)
            final_action(1, max_indices.unsqueeze(1), 1)
        return final_action
    
    def updateValue(self, action, prices) -> None:
        temp_holdings = self.holdings.clone()
        temp_holdings = temp_holdings + torch.sum((action * self.update_tensor * temp_holdings), dim=1, keepdim=True)
        
        if torch.sum(temp_holdings < 0) > 0:
            #TODO : Change so it has different recorded value to put into the csv file
            pass
        else:
            cash_change = torch.tensor(torch.sum(prices * torch.sum((action * self.update_tensor * self.holdings), dim=1, keepdim=True)))
            if cash_change <= self.cash:
                self.holdings = self.holdings + torch.sum((action * self.update_tensor * self.holdings), dim=1, keepdim=True)
                self.cash = self.cash - cash_change
        self.value = torch.sum(self.holdings * prices) + self.cash
        self.stepPerformance()
    def stepPerformance(self):
        self.performance = self.value / self.init_balance - self.performance

    def finalPerformance(self):
        return self.value / self.init_balance
    def remember(self, state, action, reward, next_state, is_done):
        self.memory.append((action, state, reward, next_state))
