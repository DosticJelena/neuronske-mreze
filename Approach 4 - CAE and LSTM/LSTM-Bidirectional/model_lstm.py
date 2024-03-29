import torch.nn as nn
import torch.nn.functional as F
from Experiments.training_config import Activations


class BidirectionalLSTM(nn.Module):
    def __init__(
            self,
            input_size=21,
            hidden_size=42,
            num_classes=5,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=84, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

        self.relu = Activations.RELU.value

    def forward(self, _input):
        x = self.lstm(_input)
        x = self.flatten(x[0])

        x = self.fc1(x)
        x = self.relu(x)

        x = F.softmax(self.fc2(x), dim=1)

        return x

