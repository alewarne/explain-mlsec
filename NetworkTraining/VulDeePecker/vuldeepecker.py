#!/usr/bin/env python3
# Copyright: 2018 Tim Dengel <t.dengel@tu-braunschweig.de>
# License: GPLv3+

import numpy as np
from config_training import *
from gensim.models.word2vec import Word2Vec
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
import json
import os
from sklearn.model_selection import train_test_split
import sys
from VuldeeDataGenerator import VuldeeDataGenerator

sys.path.append('../../utils/')
from custom_metrics import custom_true_positive_metric, custom_false_positive_metric


def load_data(gadgets, w2v):
    x = [[w2v[word] for word in gadget["tokens"]] for gadget in gadgets]
    y = [[1,0] if gadget["label"] == 0 else [0,1] for gadget in gadgets]

    types = [gadget["type"] for gadget in gadgets]
    return x, y, types


def pad_one(xi_typei):
    xi, typei = xi_typei
    if typei == 1:
        if len(xi) > token_per_gadget:
            ret = xi[0:token_per_gadget]
        elif len(xi) < token_per_gadget:
            ret = xi + [[0] * len(xi[0])] * (token_per_gadget - len(xi))
        else:
            ret = xi
    elif typei == 0 or typei == 2: # Trunc/append at the start
        if len(xi) > token_per_gadget:
            ret = xi[len(xi) - token_per_gadget:]
        elif len(xi) < token_per_gadget:
            ret = [[0] * len(xi[0])] * (token_per_gadget - len(xi)) + xi
        else:
            ret = xi
    else:
        raise Exception()

    return ret


def padding(x, types):
    return np.array([pad_one(bar) for bar in zip(x, types)])


def get_model(final_activation='softmax'):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=no_lstm_units), input_shape=(token_per_gadget, embedding_dim)))
    model.add(Dropout(dropout_proba))
    model.add(Dense(2, activation=final_activation))
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy',custom_true_positive_metric(vec_output),
                                            custom_false_positive_metric(vec_output)])
    return model


def train_model(model, training_generator, test_generator):
    if not os.path.isdir('models'):
        os.makedirs('models')
    fname = '/models/model.{epoch:03d}--ACC_{val_acc:.4f}--FP_{val_false_positive_metric:.4f}--' \
            'TP_{val_true_positive_metric:.4f}.hdf5'
    model_checkpoint_tp = ModelCheckpoint(fname, monitor='val_true_positive_metric', save_best_only=True, mode='max')
    model_checkpoint_fp = ModelCheckpoint(fname, monitor='val_false_positive_metric', save_best_only=True, mode='min')
    model.fit_generator(generator=training_generator, epochs=epochs, validation_data=test_generator, max_queue_size=10,
                        callbacks=[model_checkpoint_tp, model_checkpoint_fp])


def preprocess_data(x, y, types):
    x = padding(x, types)
    # Train/Test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testset_size, random_state=sampling_random_seed)
    y_train, y_test = np.array(y_train), np.array(y_test)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    datagen_train = VuldeeDataGenerator(x_train, y_train, batch_size)
    datagen_test = VuldeeDataGenerator(x_test, y_test, batch_size)

    return datagen_train, datagen_test


if __name__ == "__main__":
    w2v = Word2Vec.load(w2v_path)
    with open(data_path) as f:
        gadgets = json.load(f)
        x, y, types = load_data(gadgets, w2v)
        del gadgets
        del w2v
        print("Loaded data.")
    # pad sequences, split data, create datagens
    datagen_train, datagen_test = preprocess_data(x,y, types)
    vuldee_model = get_model()
    train_model(vuldee_model, datagen_train, datagen_test)
