import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from Experiments.config import Config
from Experiments.data import get_dataloader
from Experiments.training_config import Losses, Optimizers, DataLoaderMode

config = Config()


class Trainer:
    def __init__(self, model, lr, batch_size, num_epochs, model_type='cnn'):
        self.model = model.to(config.device)
        self.model_type = model_type
        self.num_epochs = num_epochs
        if model_type == 'cae':
            self.criterion = Losses.MSE.value
        else:
            self.criterion = Losses.CROSS_ENT.value
        self.optimizer = Optimizers.ADAM_W.value(self.model.parameters(), lr=lr)
        self.best_loss = float('inf')
        self.modes = [DataLoaderMode.TRAIN.value, DataLoaderMode.VALIDATION.value]
        self.dataloaders = {
            mode: get_dataloader(mode, batch_size, model_type) for mode in self.modes
        }
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()

    def _train_epoch(self, mode, model_type='cnn'):
        print("-----------\n{} mode | time: {}".format(mode, time.strftime('%H:%M:%S')))

        self.model.train() if mode == 'train' else self.model.eval()
        metrics_manager = MetricsManager()
        metrics_manager.init_metrics()
        for i, (data, target) in enumerate(self.dataloaders[mode]):
            data = data.to(config.device)
            target = target.to(config.device)
            output = self.model(data)

            if model_type == 'cae':
                loss = self.criterion(output, data)
            else:
                loss = self.criterion(output, target)

            if mode == DataLoaderMode.TRAIN.value:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            metrics_manager.update(output, target, loss.item())

        metrics = metrics_manager.get_metrics()
        metrics = {k: v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])

        if mode == DataLoaderMode.TRAIN.value:
            self.train_df_logs = pd.concat([self.train_df_logs, df_logs], axis=0)
        else:
            self.val_df_logs = pd.concat([self.val_df_logs, df_logs], axis=0)

        # show logs
        print('- {}: {}\n- {}: {}\n- {}: {}\n- {}: {}\n- {}: {}'.format(*(x for kv in metrics.items() for x in kv)))

        return loss

    def run(self):
        for epoch in range(self.num_epochs):
            self._train_epoch(mode=DataLoaderMode.TRAIN.value, model_type=self.model_type)
            with torch.no_grad():
                val_loss = self._train_epoch(mode=DataLoaderMode.VALIDATION.value, model_type=self.model_type)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('\nNew checkpoint\n')
                self.best_loss = val_loss
                #torch.save(self.model.state_dict(), f"LSTM-Bidirectional/Models/best_model_epoch{epoch}.pth")


class MetricsManager:
    def __init__(self, n_classes=5):
        self.metrics = {}
        self.confusion = torch.zeros((n_classes, n_classes))

    def init_metrics(self):
        self.metrics['loss'] = 0
        self.metrics['accuracy'] = 0
        self.metrics['f1'] = 0
        self.metrics['precision'] = 0
        self.metrics['recall'] = 0

    def get_metrics(self):
        return self.metrics

    def update(self, x, y, loss):
        x = np.argmax(x.detach().cpu().numpy(), axis=1)
        y = y.detach().cpu().numpy()
        self.metrics['loss'] += loss
        self.metrics['accuracy'] += accuracy_score(x, y)
        self.metrics['f1'] += f1_score(x, y, average='macro')
        self.metrics['precision'] += precision_score(x, y, average='macro', zero_division=1)
        self.metrics['recall'] += recall_score(x, y, average='macro', zero_division=1)