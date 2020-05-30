import numpy as np

class NeuralNetworkTrainer:
  
  reporter = lambda self, text : None
  
  def __init__(self, network):
    self.rng = np.random.default_rng()
    self.network = network
  
  def train(self, training_data, mini_batch_size, number_of_epochs, learning_rate):
    for epoch_number in range(number_of_epochs):
      self.reporter("Epoch: {0}/{1}".format(epoch_number+1, number_of_epochs))
      self.run_epoch(training_data[:], mini_batch_size, learning_rate)
  
  def run_epoch(self, training_data, mini_batch_size, learning_rate):
    training_list = list(training_data)
    self.rng.shuffle(training_list)
    training_data = tuple(training_list)
    mini_batch_start_index = 0
    while mini_batch_start_index < len(training_data):
      self.run_mini_batch(training_data[mini_batch_start_index:mini_batch_start_index+mini_batch_size], learning_rate)
      mini_batch_start_index += mini_batch_size
  
  def run_mini_batch(self, mini_batch_training_data, learning_rate):
    batch_size = len(mini_batch_training_data)
    sum_of_weights_gradients = [np.zeros(weights.shape) for weights in self.network.weights]
    sum_of_biases_gradients = [np.zeros(biases.shape) for biases in self.network.biases]
    for input, expected_output in mini_batch_training_data:
      weights_gradient, biases_gradient = self.backpropagate(input, expected_output)
      sum_of_weights_gradients = [ sum + new_values for sum, new_values in zip(sum_of_weights_gradients, weights_gradient) ]
      sum_of_biases_gradients = [ sum + new_values for sum, new_values in zip(sum_of_biases_gradients, biases_gradient) ]
    change_to_weights = [-weights/batch_size * learning_rate for weights in sum_of_weights_gradients]
    change_to_biases = [-biases/batch_size * learning_rate for biases in sum_of_biases_gradients]
    self.network.update_weights_and_biases(change_to_weights, change_to_biases)
  
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
    layer_errors = self.cost_derivative(activations[-1], expected_output) * self.network.activation_derivative(weighted_inputs[-1])
    # Go through layers backwards
    for layer in range(len(weights_gradients)-1, -1, -1):
      weights_gradients[layer] = np.outer(layer_errors, activations[layer].transpose())
      biases_gradients[layer] = layer_errors[:]
      if layer > 0:
        layer_errors = np.array(self.network.weights[layer]).transpose().dot(np.array(layer_errors)) * self.network.activation_derivative(weighted_inputs[layer-1])
    return weights_gradients, biases_gradients
  
  def cost_derivative(self, output, expected_output):
    return output - expected_output
  
  def test(self, test_data):
    correct_count = 0
    for input, expected_output in test_data:
      output = self.network.feed_forward(input)
      value = np.argmax(output)
      expected_value = np.argmax(expected_output)
      if value == expected_value:
        correct_count += 1
    return correct_count