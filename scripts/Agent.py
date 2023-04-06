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
class Agent:
    def __init__(self, cash):
        
        self.cash = torch.tensor(cash).to("cuda:0")
        self.init_balance = torch.tensor(cash).to("cuda:0")
        self.holdings_value = torch.tensor(0).to("cuda:0")
        self.value = torch.tensor(cash).to("cuda:0")
        
        self.memory = deque(maxlen=46800)
        self.holdings_average = torch.tensor(0).to("cuda:0")
        self.performance = torch.tensor(0).to("cuda:0")
        self.model = QNet(155,1000000,3).to("cuda:0")
        self.model.share_memory()
        
        self.epochs = torch.tensor(0)

    def getState(self, data):
        agent_data = torch.tensor([self.cash, self.holdings_value, 
            self.holdings_average, self.performance]).to("cuda:0")
        state = data.clone().to("cuda:0")

        state = torch.cat((state, agent_data),0)
        return state

    def getAction(self,data):
        
        self.epsilon = 100 - self.epochs
        final_action = [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]
        
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
    
    def openPosition(self, data):
        
        if(self.cash >= data[1]* 5):
            self.cash -= data[1] * 5
            self.holdings_value = self.n_stocks * data[1] + data[1] * 5
            self.n_stocks += 5
            self.holdings_average = self.holdings_value / self.n_stocks
            self.value = self.cash + self.holdings_value
            return True
        else:
            return False
    def closePosition(self, data):
        if(self.n_stocks >= 5):
            self.cash += data[1] * 5
            
            self.holdings_value = self.n_stocks * data[1] - data[1] * 5
            self.holdings_average = self.holdings_value / self.n_stocks
            self.n_stocks -= 5
            self.value = self.cash + self.holdings_value
            return True
        else:
            return False
    def calcPerformance(self):
        self.performance = self.value/self.init_balance
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))