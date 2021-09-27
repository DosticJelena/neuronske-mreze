from model_lstm import UnidirectionalLSTM
from Experiments.training_components import Trainer

model_to_train = UnidirectionalLSTM()
trainer = Trainer(model=model_to_train, lr=1e-3, batch_size=96, num_epochs=10, model_type='lstm')
trainer.run()
