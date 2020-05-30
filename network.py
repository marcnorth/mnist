import numpy as np
from enum import Enum

class NeuralNetwork:
  
  class Init(Enum):
    ZERO = 1
    NORMAL = 2
  
  def __init__(self, layer_sizes, init=Init.NORMAL):
    self.rng = np.random.default_rng()
    self.layer_sizes = layer_sizes
    self.input_size = layer_sizes[0]
    self.active_layer_sizes = layer_sizes[1:]
    self.init_weights_and_biases(init)
    
  def init_weights_and_biases(self, init):
    self.weights = [];
    self.biases = [];
    for i in range(len(self.layer_sizes)-1):
      self.weights.append(self.generate_layer_weights(i) if init == NeuralNetwork.Init.NORMAL else self.zero_layer_weights(i))
      self.biases.append(self.generate_layer_biases(i) if init == NeuralNetwork.Init.NORMAL else self.zero_layer_biases(i))
  
  def generate_layer_weights(self, layer_index):
    return self.rng.standard_normal((self.layer_sizes[layer_index+1], self.layer_sizes[layer_index]))
  
  def generate_layer_biases(self, layer_index):
    return self.rng.standard_normal(self.layer_sizes[layer_index+1])
  
  def zero_layer_weights(self, layer_index):
    return np.zeros((self.layer_sizes[layer_index+1], self.layer_sizes[layer_index]))
  
  def zero_layer_biases(self, layer_index):
    return np.zeros(self.layer_sizes[layer_index+1])
  
  def feed_forward(self, input):
    a = input
    for layer_weights, layer_biases in zip(self.weights, self.biases):
      a = self.activate(self.weighted_input(a, layer_weights, layer_biases))
    return a
  
  def weighted_input(self, input, layer_weights, layer_biases):
    return layer_weights.dot(input) + layer_biases
  
  def activate(self, z):
    return 1.0 / (1.0 + np.exp(-z))
  
  def update_weights_and_biases(self, change_to_weights, change_to_biases):
    self.weights = [layer_weights + change_to_layer_weights for layer_weights, change_to_layer_weights in zip(self.weights, change_to_weights)]
    self.biases = [layer_biases + change_to_layer_biases for layer_biases, change_to_layer_biases in zip(self.biases, change_to_biases)]
  
  def activation_derivative(self, z):
    return self.activate(z)*(1-self.activate(z))