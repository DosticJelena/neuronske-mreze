import torch.nn as nn
import torch.nn.functional as F
from Experiments.training_config import Activations


class LSTM(nn.Module):
    def __init__(
            self,
            input_size=22,
            hidden_size=32,
            num_classes=5,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

        self.relu = Activations.RELU.value

    def forward(self, _input):
        x = self.lstm(_input)
        x = self.flatten(x[0])

        x = self.fc1(x)
        x = self.relu(x)

        # x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc2(x), dim=1)

        return x

