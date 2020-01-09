import numpy as np
import keras


# simple data generator yielding batches of data from a numpy array
class VuldeeDataGenerator(keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        length = int(np.floor(len(self.data)/self.batch_size))
        return length if len(self.data)%self.batch_size == 0 else length+1

    def __getitem__(self, idx):
        data_batch = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
        label_batch = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
        return data_batch, label_batch
