from model_cae import ConvAutoencoder
from Experiments.training_components import Trainer
from Experiments.training_config import DataLoaderMode
from Experiments.config import Config
from Experiments.data import get_dataloader
import pandas as pd
import numpy as np
import torch

config = Config()

model_to_train = ConvAutoencoder()
trainer = Trainer(model=model_to_train, lr=1e-3, batch_size=96, num_epochs=10, model_type='cae')
#trainer.run()

trainer.model.load_state_dict(torch.load('CAE/Models/best_model_epoch9.pth'))

dataloader = get_dataloader(DataLoaderMode.TRAIN.value, batch_size=1)

for i, (data, target) in enumerate(dataloader):
    target = target.detach().numpy()

    data = data.to(config.device)
    encoded_data = trainer.model(data)
    encoded_data = encoded_data.view(-1, encoded_data.size(1) * encoded_data.size(2))
    encoded_data_np = encoded_data.cpu().detach().numpy()
    encoded_data_np = [np.append(encoded_data_np[0], target)]
    encoded_data_df = pd.DataFrame(encoded_data_np)

    encoded_data_df.to_csv('lstm_data.csv', mode='a', header=False, index=False)

