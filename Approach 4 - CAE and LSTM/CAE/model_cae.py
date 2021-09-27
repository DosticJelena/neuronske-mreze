import torch.nn as nn
from Experiments.training_config import Activations


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=3
        )
        self.conv_2 = nn.Conv1d(
            in_channels=16,
            out_channels=64,
            kernel_size=3
        )
        self.conv_3 = nn.Conv1d(
            in_channels=64,
            out_channels=1,
            kernel_size=3
        )
        # self.conv_4 = nn.Conv1d(
        #     in_channels=32,
        #     out_channels=1,
        #     kernel_size=3
        # )
        self.relu_1 = Activations.RELU.value
        self.relu_2 = Activations.RELU.value
        self.relu_3 = Activations.RELU.value
        self.relu_4 = Activations.RELU.value
        self.normalization_1 = nn.BatchNorm1d(num_features=64)

        self.pool_1 = nn.MaxPool1d(kernel_size=2)
        self.pool_2 = nn.MaxPool1d(kernel_size=2)
        self.pool_3 = nn.MaxPool1d(kernel_size=2)

    def forward(self, _input):
        conv1 = self.conv_1(_input)
        x = self.relu_1(conv1)
        x = self.pool_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.normalization_1(x)
        x = self.pool_2(x)

        x = self.conv_3(x)
        x = self.relu_3(x)

        # x = self.conv_4(x)
        # x = self.relu_4(x)

        x = self.pool_3(x)

        return x


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        # self.conv_t_4 = nn.ConvTranspose1d(
        #     in_channels=1,
        #     out_channels=1,
        #     kernel_size=3,
        #     stride=2
        # )
        self.conv_t_3 = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2
        )
        self.conv_t_2 = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=2
        )
        self.conv_t_1 = nn.ConvTranspose1d(
            in_channels=64,
            out_channels=16,
            kernel_size=3,
            stride=2
        )
        self.relu_4 = Activations.RELU.value
        self.relu_3 = Activations.RELU.value
        self.relu_2 = Activations.RELU.value
        self.relu_1 = Activations.RELU.value

        self.pool_3 = nn.MaxPool1d(kernel_size=2)
        self.pool_2 = nn.MaxPool1d(kernel_size=2)
        self.pool_1 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=2800, out_features=187)

        self.sigmoid = Activations.SIGMOID.value

    def forward(self, _input):
        # conv_t_4 = self.conv_t_4(_input)
        # x = self.relu_4(conv_t_4)

        x = self.conv_t_3(_input)
        x = self.relu_3(x)

        x = self.conv_t_2(x)
        x = self.relu_2(x)

        x = self.conv_t_1(x)
        x = self.relu_1(x)

        x = self.flatten(x)
        x = self.linear(x)

        x = self.sigmoid(x)

        return x


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, _input):
        encoded = self.encoder(_input)
        decoded = self.decoder(encoded)
        return decoded

