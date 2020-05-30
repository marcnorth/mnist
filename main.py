import numpy as np
import time

from network import NeuralNetwork
from mnist_loader import MnistLoader

NeuralNetwork.reporter = print

network = NeuralNetwork([784, 30, 10])

loader = MnistLoader("mnist.pkl.gz")
training_data = loader.get_training_data().get_zipped_data()
validation_data = loader.get_validation_data().get_zipped_data()

print("{0}/{1}".format(network.test(validation_data), len(validation_data)))

network.stochastic_gradient_decent(
  training_data,
  30,
  10,
  3.0
)

print("{0}/{1}".format(network.test(validation_data), len(validation_data)))