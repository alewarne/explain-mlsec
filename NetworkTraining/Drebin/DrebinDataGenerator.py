import keras
import random
import numpy as np
import os


# data generator that takes as input a list of filenames. The generator yields batches of indices indicating which
# features are set to one during training
class DrebinDataGenerator(keras.utils.Sequence):
    def __init__(self, vec, data_list, feature_path, label_dict, batch_size, shuffle=True, vec_labels=True):
        self.data_names = data_list
        self.data_paths = [os.path.join(feature_path, item) for item in data_list]
        self.label_dict = label_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.vec = vec
        self.data = vec.transform(self.data_paths)
        self.vec_labels = vec_labels
        self.on_epoch_end()
    '''
    shuffle data names after epoch to have different batches every iteration. we have to update the data matrix
    aswell to know which samples are in which row
    '''
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data_paths)
            self.data = self.vec.transform(self.data_paths)

    def __len__(self):
        return int(np.floor(len(self.data_names)/self.batch_size))

    # returns label representation for binary labels.
    def label_to_representation(self, label):
        if self.vec_labels:
            if label == 0:
                return [1, 0]
            else:
                return [0, 1]
        else:
            return label

    def __getitem__(self, idx):
        data_batch = self.data[idx*self.batch_size:(idx+1)*self.batch_size, :].toarray()
        labels = [self.label_dict[os.path.split(path)[1]] for path in self.data_paths[idx*self.batch_size:
                                                                                      (idx + 1)*self.batch_size]]
        label_batch = np.array([self.label_to_representation(label) for label in labels], dtype='uint8')
        return data_batch, label_batch
