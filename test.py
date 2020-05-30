import unittest
import numpy as np
from network import NeuralNetwork
from network_trainer import NeuralNetworkTrainer

class TestStringMethods(unittest.TestCase):
  
  def test_init_shape(self):
    network = NeuralNetwork((3, 4, 5))
    self.assertEqual(len(network.weights), 2)
    self.assertEqual(len(network.biases), 2)
    self.assertEqual(network.weights[0].shape, (4, 3))
    self.assertEqual(network.weights[1].shape, (5, 4))
    self.assertEqual(network.biases[0].shape, (4,))
    self.assertEqual(network.biases[1].shape, (5,))
  
  def test_init_zero(self):
    network = NeuralNetwork((3, 4, 5), init=NeuralNetwork.Init.ZERO)
    self.assertTrue(np.array_equal(network.weights[0], np.array(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)))))
  
  def test_output(self):
    network = NeuralNetwork((3, 4, 5), init=NeuralNetwork.Init.ZERO)
    change_to_weights = [
      np.array([
        [0.3, 0. , 0. ],
        [0. , 0. , 0. ],
        [0. , 0. , 0. ],
        [0. , 0. , 0. ]
      ]),
      np.array([
        [0.4, 0. , 0. , 0. ],
        [0.9, 0. , 0. , 0. ],
        [0. , 0. , 0. , 0. ],
        [0. , 0. , 0. , 0. ],
        [0. , 0. , 0. , 0. ]
      ])
    ]
    change_to_biases = [
      np.array([0., 0., 0., 0.]),
      np.array([0. , 0. , 0. , 0. , 0.9])
    ]
    network.update_weights_and_biases(change_to_weights, change_to_biases)
    self.assertTrue(np.array_equal(network.feed_forward((-2, -4, -6)), np.array((0.5353751667092154, 0.5790584231739951, 0.5, 0.5, 0.7109495026250039))))
  
  def test_trainer(self):
    network = NeuralNetwork((3, 10, 8), init=NeuralNetwork.Init.ZERO)
    trainer = NeuralNetworkTrainer(network)
    training_data = [
      (np.array((0, 0, 0)), np.array((1, 0, 0, 0, 0, 0, 0, 0))),
      (np.array((0, 0, 1)), np.array((0, 1, 0, 0, 0, 0, 0, 0))),
      (np.array((0, 1, 0)), np.array((0, 0, 1, 0, 0, 0, 0, 0))),
      (np.array((0, 1, 1)), np.array((0, 0, 0, 1, 0, 0, 0, 0))),
      (np.array((1, 0, 0)), np.array((0, 0, 0, 0, 1, 0, 0, 0))),
      (np.array((1, 0, 1)), np.array((0, 0, 0, 0, 0, 1, 0, 0))),
      (np.array((1, 1, 0)), np.array((0, 0, 0, 0, 0, 0, 1, 0))),
      (np.array((1, 1, 1)), np.array((0, 0, 0, 0, 0, 0, 0, 1)))
    ]
    test_data = training_data[:]
    self.assertEqual(1, trainer.test(test_data))
    trainer.train(training_data, mini_batch_size=2, number_of_epochs=3000, learning_rate=5.0)
    self.assertEqual(8, trainer.test(test_data))
    
if __name__ == '__main__':
    unittest.main()