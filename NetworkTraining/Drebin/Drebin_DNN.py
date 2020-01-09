import pickle
import os
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from DrebinDataGenerator import DrebinDataGenerator
from drebin_datapipeline import virustotal_json_to_labels, get_train_test_valid_names, get_count_vectorizer
from config import *

sys.path.append('../../utils/')
from custom_metrics import custom_true_positive_metric, custom_false_positive_metric

# the network used by grosse et. al in the paper 'adversarial examples for malware detection'
def get_network(no_features, final_nonlinearity, vec_output):
    model = Sequential()
    model.add(Dense(units=200, activation='relu', input_shape=(no_features, )))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=200, activation='relu'))
    model.add(Dropout(rate=0.5))
    if vec_output:
        model.add(Dense(units=2, activation=final_nonlinearity))
    else:
        model.add(Dense(units=1, activation=final_nonlinearity))
    return model


# train the model
def train_model(model, training_gen, test_gen, loss, epochs, vec_output, save_period=1):
    print(model.summary())
    if not os.path.isdir('models'):
        os.makedirs('models')
    model.compile(optimizer=SGD(), loss=loss, metrics=['accuracy', custom_true_positive_metric(vec_output),
                                                       custom_false_positive_metric(vec_output)])
    fname = 'models/model.{epoch:03d}--ACC_{val_accuracy:.4f}--FP_{val_false_positive_metric:.4f}--' \
            'TP_{val_true_positive_metric:.4f}.hdf5'
    model_checkpoint_tp = ModelCheckpoint(fname, monitor='true_positive_metric', save_best_only=True, mode='max',
                                          period=save_period)
    model_checkpoint_fn = ModelCheckpoint(fname, monitor='val_false_positive_metric', save_best_only=True, mode='min',
                                          period=save_period)
    model.fit_generator(generator=training_gen, epochs=epochs, class_weight={0:1, 1:6.5}, validation_data=test_gen,
                        max_queue_size=10, callbacks=[model_checkpoint_tp, model_checkpoint_fn])


# returns two lists of filenames, one for training and one for testing. The lists are specified by a doc path (to a file
# containing the feature vectors) and a split path containing files with filenames for training, testing, validation.
# since there are several splits for the dataset, the index spcifies which split to choose
def get_train_test_data_names(split_path, index, label_dict):
    names = get_train_test_valid_names(split_path)
    train_names = names[index][0]
    train_names = [name for name in train_names if name in label_dict]
    test_names = names[index][1]
    test_names = [name for name in test_names if name in label_dict]
    # val_names = names[0][2]
    return train_names, test_names


if __name__ == '__main__':
    if not os.path.isfile('train_label_dict.pkl'):
        print('Calculating label dict ...')
        label_dict = virustotal_json_to_labels(json_path, threshold)
        pickle.dump(label_dict, open('train_label_dict.pkl', 'wb'))
    else:
        label_dict = pickle.load(open('train_label_dict.pkl', 'rb'))
    if not os.path.isfile('train_vec.pkl'):
        print('Calculating count vectorizer ...')
        vec = get_count_vectorizer(doc_path, label_dict)
        pickle.dump(vec, open('train_vec.pkl', 'wb'))
    else:
        vec = pickle.load(open('train_vec.pkl', 'rb'))
    no_tokens = len(vec.vocabulary_)
    train_data_names, test_data_names = get_train_test_data_names(split_path, split_index, label_dict)
    train_data_gen = DrebinDataGenerator(vec, train_data_names, doc_path, label_dict, batch_size, vec_labels=vec_output)
    test_data_gen = DrebinDataGenerator(vec, test_data_names, doc_path, label_dict, batch_size, vec_labels=vec_output)
    model = get_network(no_tokens, final_nonlinearity=nonlinearity, vec_output=vec_output)
    print('training with %d tokens' % no_tokens)
    train_model(model, train_data_gen, test_data_gen, loss, epochs=epochs, vec_output=vec_output)
