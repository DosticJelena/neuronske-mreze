import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from Experiments.config import Config, DataPaths
from Experiments.training_config import DataLoaderMode

config = Config()


class MITBIHDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.data_columns = self.df.columns[:-1].tolist()

    def __getitem__(self, index):
        x = self.df.iloc[index, :-1].astype('float')
        y = self.df.iloc[index, -1]
        x = torch.FloatTensor([x.values])
        y = torch.LongTensor(np.array(y))
        return x, y

    def __len__(self):
        return len(self.df)


class LSTMDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.data_columns = self.df.columns[:-1].tolist()

    def __getitem__(self, index):
        x = self.df.iloc[index, :-1].astype('float')
        y = self.df.iloc[index, -1]
        x = torch.FloatTensor([x.values])
        y = torch.LongTensor(np.array(y))
        return x, y

    def __len__(self):
        return len(self.df)


def get_dataloader(mode: DataLoaderMode, batch_size=96, model_type='cnn'):
    if model_type == 'lstm':
        df = pd.read_csv(DataPaths.LSTM_DATA_ENCODED.value, header=None)
        dataset = LSTMDataset(df)
    else:
        df = pd.read_csv(DataPaths.MIT_TRAIN.value, header=None)

        train_df, val_df = train_test_split(
            df,
            test_size=0.15,
            random_state=config.seed,
            stratify=df.iloc[:, -1])  # stratify - not balanced dataset
        df = train_df if mode == DataLoaderMode.TRAIN.value else val_df
        dataset = MITBIHDataset(df)

    return DataLoader(dataset=dataset, batch_size=batch_size)


