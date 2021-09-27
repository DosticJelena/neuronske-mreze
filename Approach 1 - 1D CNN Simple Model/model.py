import torch.nn as nn
import torch.nn.functional as F
from Experiments.training_config import Activations


class ConvNormPool(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            kernel_size
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.tanh_1 = Activations.TANH.value
        self.tanh_2 = Activations.TANH.value
        self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
        self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)

        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, _input):
        conv1 = self.conv_1(_input)
        x = self.normalization_1(conv1)
        x = self.tanh_1(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.tanh_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.pool(x)

        return x


class CNN(nn.Module):
    def __init__(
            self,
            input_size=1,
            hid_size=256,
            kernel_size=5,
            num_classes=5,
    ):
        super().__init__()

        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size // 2,
            kernel_size=kernel_size,
        )
        self.conv3 = ConvNormPool(
            input_size=hid_size // 2,
            hidden_size=hid_size // 4,
            kernel_size=kernel_size,
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))

        self.fc = nn.Linear(in_features=hid_size // 4, out_features=num_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)

        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)
        return x

