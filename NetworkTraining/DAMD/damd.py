import os
import sys
import argparse
from tqdm import tqdm
import pickle as pkl
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv1D, Embedding, GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from config import *

sys.path.append('../../utils/')
from custom_metrics import custom_true_positive_metric, custom_false_positive_metric


# sorts a list of filenames in uprising order with respect to the number of tokens in the files
def get_sorted_list_of_filenames(data_path):
    fname_list = os.listdir(data_path)
    fname_to_len = {}
    print('Sorting input by length ...')
    for fname in tqdm(fname_list):
        fname_to_len[fname] = len(open(os.path.join(data_path, fname), 'r').read().split(','))
    sorted_fname = sorted(fname_to_len.items(), key=lambda kv: kv[1])
    return [tup[0] for tup in sorted_fname]


# turns a (sorted) list of filenames of indices into two numpy arrays. One containing the indices and the other one
# containing the labels. Assumes that the last part of the filename is '.1' for malicious and '.0' for benign
def filename_list_to_numpy_arrays(filenames, root_path):
    indices = []
    # labels are either [1,0] or [0,1]
    labels = np.zeros(shape=(len(filenames), 2))
    for i, filename in enumerate(filenames):
        full_path = os.path.join(root_path, filename)
        with open(full_path, 'r') as f:
            indices.append(np.array(f.read().split(','), dtype=np.uint8))
        labels[i,:] = [1,0] if filename.split('.')[-1] == '0' else [0,1]
    return np.array(indices), labels


def get_damd_cnn(no_tokens, final_nonlinearity='softmax'):
    embedding_dimensions = 8
    no_convolutional_filters = 64
    number_of_dense_units = 16
    kernel_size = 8
    no_labels = 2
    model = Sequential()
    model.add(Embedding(input_dim=no_tokens+1, output_dim=embedding_dimensions))
    model.add(Conv1D(filters=no_convolutional_filters, kernel_size=kernel_size, padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(number_of_dense_units, activation='relu'))
    model.add(Dense(no_labels, activation=final_nonlinearity))
    print(model.summary())
    return model


def train_network_batchwise(data_path, network, no_epochs, batch_size, testset_size, random_state=42):
    if not os.path.isdir('models'):
        os.makedirs('models')
    network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',
                                                                              custom_true_positive_metric(vec_output),
                                                                              custom_false_positive_metric(vec_output)
                                                                              ])
    filenames_sorted = get_sorted_list_of_filenames(data_path)
    names_train, names_test = train_test_split(filenames_sorted, test_size=testset_size, random_state=random_state)
    for j in range(no_epochs):
        acc_train, acc_test = [], []
        print('Training epoch {}'.format(j+1))
        for i in tqdm(range(0, len(names_train), batch_size)):
            x, y = filename_list_to_numpy_arrays(names_train[i:i+batch_size], data_path)
            res = network.train_on_batch(pad_sequences(x, len(x[-1]), dtype='uint8'), y)
            acc_train.append(res[1])
        print('Train accuracy after {} epochs: {}:'.format(j+1, np.mean(acc_train)))
        for k in range(0, len(names_test), batch_size):
            x, y = filename_list_to_numpy_arrays(names_test[k:k+batch_size], data_path)
            res = network.test_on_batch(pad_sequences(x, len(x[-1]), dtype='uint8'), y)
            acc_test.append(res[1])
        print('Test accuracy after {} epochs: {}:'.format(j+1, np.mean(acc_test)))
        network.save('models/damd_model_%d' % j)


if __name__ == '__main__':
    damd_model = get_damd_cnn(no_tokens)
    train_network_batchwise(token_path, damd_model, epochs, batch_size, testset_size)