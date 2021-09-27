from model import CNN
from Experiments.training_components import Trainer


model_to_train = CNN(num_classes=5, hid_size=128)
trainer = Trainer(model=model_to_train, lr=1e-3, batch_size=96, num_epochs=10)
trainer.run()
