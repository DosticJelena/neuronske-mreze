from model_lstm import BidirectionalLSTM
from Experiments.training_components import Trainer

model_to_train = BidirectionalLSTM()
trainer = Trainer(model=model_to_train, lr=1e-3, batch_size=96, num_epochs=10, model_type='lstm')
trainer.run()
