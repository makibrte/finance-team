import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
import transformScript as tr

ffile_path = '../Data/planets_txt/'
file_path_export = '../Data/planets_csv/'
tensor_list = []

def main():
    for x, file in tqdm(enumerate(os.listdir(file_path_export))):
        df = tr.read_csv_transform('{}{}'.format(file_path_export, file), 'd').drop(['Date__(UT)__HR:MN', 'date'], axis=1)
        # create a new DataFrame with 100 columns of all zeros
        df = df.dropna(axis=0)
        print(file)
        print(df.shape)
        dummy_df = pd.DataFrame(np.zeros((df.shape[0], 78)), index=df.index)

    # set column names for the dummy DataFrame
        dummy_df.columns = ['dummy_' + str(i) for i in range(78)]

    # concatenate the dummy DataFrame with the original DataFrame
        df = pd.concat([df, dummy_df], axis=1)

        tens = torch.tensor(df.values)
        tensor_list.append(tens)
        print(tens.shape)

# if name == 'main':
if __name__ == '__main__':

    main()
