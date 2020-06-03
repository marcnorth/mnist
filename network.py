import numpy as np
from enum import Enum
from helpers import sigmoid

class NeuralNetwork:
  
  class Init(Enum):
    ZERO = 1
    NORMAL = 2
    SMALL = 3
  
  def __init__(self, layer_sizes, init=Init.SMALL):
    self.layer_sizes = layer_sizes
    self.input_size = layer_sizes[0]
    self.active_layer_sizes = layer_sizes[1:]
    self.init_weights_and_biases(init)
    
  def init_weights_and_biases(self, init):
    self.weights = [];
    self.biases = [];
    for i in range(len(self.layer_sizes)-1):
      if init == NeuralNetwork.Init.ZERO:
        self.weights.append(np.zeros((self.layer_sizes[i+1], self.layer_sizes[i])))
        self.biases.append(np.zeros(self.layer_sizes[i+1]))
      elif init == NeuralNetwork.Init.NORMAL:
        self.weights.append(np.random.default_rng().standard_normal((self.layer_sizes[i+1], self.layer_sizes[i])))
        self.biases.append(np.random.default_rng().standard_normal(self.layer_sizes[i+1]))
      elif init == NeuralNetwork.Init.SMALL:
        self.weights.append(np.random.default_rng().normal(scale=1/np.sqrt(self.layer_sizes[i]), size=(self.layer_sizes[i+1], self.layer_sizes[i])))
        self.biases.append(np.random.default_rng().normal(size=self.layer_sizes[i+1]))
      
  def feed_forward(self, input):
    a = input
    for layer_weights, layer_biases in zip(self.weights, self.biases):
      a = self.activate(self.weighted_input(a, layer_weights, layer_biases))
    return a
  
  def weighted_input(self, input, layer_weights, layer_biases):
    return layer_weights.dot(input) + layer_biases
  
  def activate(self, z):
    return sigmoid(z)
  
  def update_weights_and_biases(self, change_to_weights, change_to_biases, weight_decay_rate=1):
    self.weights = [weight_decay_rate * layer_weights + change_to_layer_weights for layer_weights, change_to_layer_weights in zip(self.weights, change_to_weights)]
    self.biases = [layer_biases + change_to_layer_biases for layer_biases, change_to_layer_biases in zip(self.biases, change_to_biases)]
  
  def activation_derivative(self, z):
    return self.activate(z)*(1-self.activate(z))