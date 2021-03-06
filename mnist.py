from network import NeuralNetwork
from network_trainer import NeuralNetworkTrainer
from mnist_loader import MnistLoader
import network_trainer

NeuralNetworkTrainer.reporter = print

network = NeuralNetwork([784, 30, 10])
trainer = NeuralNetworkTrainer(network)

loader = MnistLoader("mnist.pkl.gz")
training_data = loader.get_training_data().get_zipped_data()
validation_data = loader.get_validation_data().get_zipped_data()
test_data = loader.get_test_data().get_zipped_data()

trainer.train(
  training_data,
  mini_batch_size=30,
  max_number_of_epochs=None,
  learning_rate=0.5,
  learning_rate_decay=0.5,
  weight_decay=5.0,
  validation_data=validation_data,
  epochs_per_validation=1,
  max_epochs_since_improvement=10
)

print("{0}/{1}".format(trainer.test(test_data), len(test_data)))