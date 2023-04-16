import os 
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

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
    torch.set_default_device(args.device)
    dataset_temp = dataset.unbind(0)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print('Works0')
    #Backtest loop
    for epoch in tqdm(range(1, 1000)):
        print('Works')
        for data in dataset_temp[:-1680]:
            
            state = agent.getState(data[-20:])
            actions = agent.getAction(state)
            agent.updateValue(actions, data[-20:])
            agent.stepPerformance()
            new_state = agent.getState(data)
            #TODO : Modify the reward function
            reward = agent.stepPerformance()
            agent.remember(actions, state, reward, new_state)
        perfromance = agent.finalPerformance()
        
        if(perfromance > record):
            record = perfromance
            agent.model.save()
            agent.remember(state, actions, perfromance, new_state)
        else:
            state = agent.getState(dataset[-1680][-20:])
            
            agent.remember(state, actions, perfromance, new_state)
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
    test(args, agent, model, device, dataset[-1680:])
    df.to_csv(index=False)
    df.to_csv(args.save_file)

def test(args, agent, model, device, dataset, dataloader_kwargs):
    for data in dataset:
        state = agent.getState(data)
        actions = agent.getAction(state)
        agent.updateValue(actions, data[-20:])
        agent.calcPerformance()
        new_state = agent.getState(data)
            #TODO : Modify the reward function
        reward = agent.stepPerformance()
        agent.remember(state, actions, agent.performance, new_state)
    perfromance = agent.finalPerformance()
    print('Model test performance')

def train_epoch(epoch, args, model, agent, device, data_loader, optimizer):
    
    model.train()
    pid = os.getpid()
    actions, states, rewards, next_states = zip(*data_loader)
    
    
    output = model(states[1])
    target = output.clone()
    for batch_idx in range(len(states)):
        
        
        Q_new = rewards[batch_idx] + 0.9 * torch.max(model(next_states[batch_idx]))
        target[batch_idx][torch.argmax(actions[batch_idx]).item()] = Q_new
    optimizer.zero_grad()
    criterion = nn.MSELoss()
    loss = criterion(output.to(device), target.to(device))
    loss.backward()
    optimizer.step()
def test_epoch(model):
    model.eval()
