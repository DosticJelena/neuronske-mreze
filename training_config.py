import torch
import torch.optim as optim
import torch.nn as nn
from enum import Enum


class Optimizers(Enum):
    SGD = optim.SGD
    ADAM = optim.Adam
    ADAM_W = optim.AdamW
    ADADELTA = optim.Adadelta
    RMSPROP = optim.RMSprop


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Activations(Enum):
    SIGMOID = nn.Sigmoid()
    TANH = nn.Tanh()
    RELU = nn.ReLU()
    L_RELU = nn.LeakyReLU()
    ELU = nn.ELU()
    SELU = nn.SELU()
    SWISH = Swish()


class Losses(Enum):
    CROSS_ENT = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()


class DataLoaderMode(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'