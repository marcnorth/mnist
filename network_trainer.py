import numpy as np
from helpers import sigmoid_prime


class QuadraticCost:
  
  @staticmethod
  def cost(output, expected_output):
    return 0.5 * np.linalg.norm(output - expected_output)**2
  
  @staticmethod
  def output_layer_error(output, expected_output, output_layer_weighted_input):
    return (output - expected_output) * sigmoid_prime(output_layer_weighted_input)


class CrossEntropyCost:
  
  @staticmethod
  def cost(output, expected_output):
    return np.sum(expected_output * np.log(output) + (1-expected_output) * np.log(1 - output))
  
  @staticmethod
  def output_layer_error(output, expected_output, output_layer_weighted_input):
    return output - expected_output


class NeuralNetworkTrainer:
  
  reporter = lambda self, text : None
  
  def __init__(self, network, cost=CrossEntropyCost):
    self.rng = np.random.default_rng()
    self.network = network
    self.cost = cost
  
  def train(self, training_data, mini_batch_size,
    learning_rate,
    learning_rate_decay=1,
    weight_decay=0,
    validation_data=None,
    max_number_of_epochs=None,
    epochs_per_validation=1,
    max_epochs_since_improvement=None
  ):
    best_score = 0
    epochs_since_improvement = 0
    epoch_number = 0
    while True:
      epoch_number += 1
      self.reporter("Epoch: {0}{1}".format(epoch_number, "/"+str(max_number_of_epochs) if max_number_of_epochs is not None else ""))
      self.run_epoch(training_data[:], mini_batch_size, learning_rate, weight_decay)
      if validation_data is not None and (epoch_number) % epochs_per_validation == 0:
        correct_count = self.validate(validation_data)
        self.reporter("After {0} epoch(s): {1}/{2}".format(epoch_number, correct_count, len(validation_data)))
        if correct_count > best_score:
          best_score = correct_count
          epochs_since_improvement = 0
        else:
          epochs_since_improvement += epochs_per_validation
          learning_rate *= learning_rate_decay
          self.reporter("No improvements for {0} epoch(s)".format(epochs_since_improvement))
        if max_epochs_since_improvement is not None and max_epochs_since_improvement <= epochs_since_improvement:
          break
      if max_number_of_epochs is not None and epoch_number >= max_number_of_epochs:
        break
  
  def run_epoch(self, training_data, mini_batch_size, learning_rate, weight_decay=0):
    training_list = list(training_data)
    self.rng.shuffle(training_list)
    training_data = tuple(training_list)
    weight_decay_rate = 1 - learning_rate * weight_decay / len(training_data)
    mini_batch_start_index = 0
    while mini_batch_start_index < len(training_data):
      self.run_mini_batch(training_data[mini_batch_start_index:mini_batch_start_index+mini_batch_size], learning_rate, weight_decay_rate)
      mini_batch_start_index += mini_batch_size
  
  def run_mini_batch(self, mini_batch_training_data, learning_rate, weight_decay_rate=1):
    batch_size = len(mini_batch_training_data)
    sum_of_weights_gradients = [np.zeros(weights.shape) for weights in self.network.weights]
    sum_of_biases_gradients = [np.zeros(biases.shape) for biases in self.network.biases]
    for input, expected_output in mini_batch_training_data:
      weights_gradient, biases_gradient = self.backpropagate(input, expected_output)
      sum_of_weights_gradients = [ sum + new_values for sum, new_values in zip(sum_of_weights_gradients, weights_gradient) ]
      sum_of_biases_gradients = [ sum + new_values for sum, new_values in zip(sum_of_biases_gradients, biases_gradient) ]
    change_to_weights = [-weights/batch_size * learning_rate for weights in sum_of_weights_gradients]
    change_to_biases = [-biases/batch_size * learning_rate for biases in sum_of_biases_gradients]
    self.network.update_weights_and_biases(change_to_weights, change_to_biases, weight_decay_rate)
  
  def backpropagate(self, input, expected_output):
    weights_gradients = [np.zeros(weights.shape) for weights in self.network.weights]
    biases_gradients = [np.zeros(biases.shape) for biases in self.network.biases]
    layer_input = input
    weighted_inputs = []
    activations = [input]
    for layer_weights, layer_biases in zip(self.network.weights, self.network.biases):
      weighted_input = self.network.weighted_input(layer_input, layer_weights, layer_biases)
      activation = self.network.activate(weighted_input)
      weighted_inputs.append(weighted_input)
      activations.append(activation)
      layer_input = activation
    # Calculate gradients for output layer
    layer_errors = self.cost.output_layer_error(activations[-1], expected_output, weighted_inputs[-1])
    # Go through layers backwards
    for layer in range(len(weights_gradients)-1, -1, -1):
      weights_gradients[layer] = np.outer(layer_errors, activations[layer].transpose())
      biases_gradients[layer] = layer_errors[:]
      if layer > 0:
        layer_errors = np.array(self.network.weights[layer]).transpose().dot(np.array(layer_errors)) * self.network.activation_derivative(weighted_inputs[layer-1])
    return weights_gradients, biases_gradients
  
  def validate(self, validation_data):
    return self.test(validation_data)
  
  def test(self, test_data):
    correct_count = 0
    for input, expected_output in test_data:
      output = self.network.feed_forward(input)
      value = np.argmax(output)
      expected_value = np.argmax(expected_output)
      if value == expected_value:
        correct_count += 1
    return correct_count