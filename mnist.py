from network import NeuralNetwork
from network_trainer import NeuralNetworkTrainer
from mnist_loader import MnistLoader

NeuralNetworkTrainer.reporter = print

network = NeuralNetwork([784, 30, 10])
trainer = NeuralNetworkTrainer(network)

loader = MnistLoader("mnist.pkl.gz")
training_data = loader.get_training_data().get_zipped_data()
validation_data = loader.get_validation_data().get_zipped_data()

print("{0}/{1}".format(trainer.test(validation_data), len(validation_data)))

trainer.train(
  training_data,
  mini_batch_size=30,
  number_of_epochs=10,
  learning_rate=3.0
)

print("{0}/{1}".format(trainer.test(validation_data), len(validation_data)))