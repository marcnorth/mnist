import pickle
import gzip

class MnistLoader:
  def __init__(self, file_path):
    file = gzip.open(file_path, 'rb')
    unpickler = pickle._Unpickler(file)
    unpickler.encoding = 'latin1'
    training_data, validation_data, test_data = unpickler.load()
    self.training_data = MnistDataSet(training_data[0], training_data[1])
    self.validation_data = MnistDataSet(validation_data[0], validation_data[1])
    self.test_data = MnistDataSet(test_data[0], test_data[1])
    file.close()
  
  def get_training_data(self):
    return self.training_data
  
  def get_validation_data(self):
    return self.validation_data
  
  def get_test_data(self):
    return self.test_data



class MnistDataSet:
  def __init__(self, image_data, correct_values):
    self.image_data = image_data
    self.correct_values = correct_values

  def get_zipped_data(self):
    return list(zip(self.image_data, self.ints_to_arrays(self.correct_values)))
  
  def ints_to_arrays(self, ints):
    arrays = []
    for value in ints:
      arrays.append([1 if value == i else 0 for i in range(0, 10)])
    return arrays