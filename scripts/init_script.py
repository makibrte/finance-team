import torch
import numpy as np
import pandas as pd

from torch import Tensor
from sklearn.preprocessing import StandardScaler

def init_tensor(filename: str):

    df = pd.read_csv(filename, index_col=0)
    tensor = torch.tensor(df.values).to('cuda:0')
    return tensor
    

def main():
    init_tensor('scaled.csv')


if __name__ == '__main__':
    main()
    
