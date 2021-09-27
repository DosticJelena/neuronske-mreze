import numpy as np
import torch
from enum import Enum
import random
import os


class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 2021

    def seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)


class DataPaths(Enum):
    MIT_TRAIN = os.path.join(os.path.dirname(__file__), "../Datasets/mitbih_train.csv")
    MIT_TEST = os.path.join(os.path.dirname(__file__), "../Datasets/mitbih_test.csv")
    LSTM_DATA_ENCODED = os.path.join(os.path.dirname(__file__), "../Datasets/lstm_data.csv")



