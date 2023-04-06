import os 
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn


def train(rank, args, model, agent, device, dataset, dataloader_kwargs):
    """
    Runs the backtesting of historical stock data
    rank : ???
    args : running paramaters
    model : Neural Network
    agent : Actor for the model
    device : CPU, GPU, MPS(Mac Only)
    dataset : Stock historical data
    dataloader_kwargs : Determines how training data(memory) is batched,
    In our case since it is Reinforcment Learning were order of data matters not recommended
    to use shuffle:True
    """
    record = 1
    torch.manual_seed(args.seed + rank)
    df = pd.DataFrame()
    n_list = []
    p_list = []
    

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #Backtest loop
    for epoch in range(1, args.epochs + 1):
        for data in dataset:
            
            state = agent.getState(data)
            actions = agent.getAction(state)
            for action in actions:
                if(action.index(max(action)) == 0):
                    agent.openPosition(data)
                elif(action.index(max(action)) == 1):
                    agent.closePosition(data)
                else:
                    agent.holdings_value = agent.n_stocks * data[1]
                    agent.value = agent.cash + agent.holdings_value
            agent.calcPerformance()
            new_state = agent.getState(data)
            agent.remember(state, action, 0, new_state, False)
        perfromance = agent.performance
        
        if(perfromance > record):
            record = perfromance
            agent.model.save()
            agent.remember(state, action, perfromance + 1, new_state, True)
        else:
            agent.remember(state, action, perfromance, new_state, True)
        #train_loader = torch.utils.data.DataLoader(agent.memory, **dataloader_kwargs)
        train_epoch(epoch, args, model, agent, device, agent.memory, optimizer)
        print(epoch)
        print(agent.value)
        #print(agent)
        n_list.append(epoch)
        p_list.append(perfromance)
        print(perfromance)
        agent.epochs += 1
        agent.cash = 10000
        agent.holdings_value = 0
        agent.n_stocks = 0
        agent.holdings_average = 0
        agent.value = 10000
        agent.performance = 1
        print(epoch)
    df['iterations'] = n_list
    df['Performance'] = p_list
    df.to_csv(index=False)
    df.to_csv('/home/matejam/Documents/IMG/IMG/backtesting/data/model-results/boreAMD.csv')

def test(args, model, device, dataset, dataloader_kwargs):
    pass

def train_epoch(epoch, args, model, agent, device, data_loader, optimizer):
    
    model.train()
    pid = os.getpid()
    states, actions, rewards, next_states, dones = zip(*data_loader)
    
    states = torch.tensor(states, dtype=torch.float, device = torch.device("cuda:0"))#
    next_states = torch.tensor(next_states, dtype=torch.float,device = torch.device("cuda:0"))#
        
    actions = torch.tensor(actions, dtype=torch.long,device = torch.device("cuda:0"))#
    rewards = torch.tensor(rewards, dtype=torch.float, device = torch.device("cuda:0"))#
    output = model(states)
    target = output.clone()
    for batch_idx in range(len(dones)):
        Q_new = rewards[batch_idx]
        if not dones[batch_idx]:
                Q_new = rewards[batch_idx] + 0.9 * torch.max(model(next_states[batch_idx]))
        target[batch_idx][torch.argmax(actions[batch_idx]).item()] = Q_new
    optimizer.zero_grad()
    criterion = nn.MSELoss()
    loss = criterion(output.to(device), target.to(device))
    loss.backward()
    optimizer.step()