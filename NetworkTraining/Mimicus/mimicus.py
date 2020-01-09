import sys
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from config import *

sys.path.append('../../utils/')
from custom_metrics import custom_true_positive_metric, custom_false_positive_metric


def get_train_data_test_data(random_seed, binary_encoding=True):
    non_relevant_columns = [1]  #filename
    label_column = 0
    arr = np.genfromtxt(path_to_csv, dtype=str, delimiter=',', skip_header=0)
    filenames = arr[1:, 1]
    no_features = arr.shape[1]
    columns_to_use = [i for i in range(no_features) if i not in non_relevant_columns]
    # feature_names = np.genfromtxt(path_to_csv, dtype=str, delimiter=',', skip_footer=9999, usecols=columns_to_use)[1:]
    # idx_to_token = dict(zip(range(len(feature_names)), feature_names))
    # pkl.dump(idx_to_token, open('data_mimicus/idx_to_token.pkl', 'wb'))
    arr = np.genfromtxt(path_to_csv, dtype=np.float, delimiter=',', skip_header=1, usecols=columns_to_use)
    labels = arr[:, label_column]
    labels = np.array([[1,0] if l == 0 else [0,1] for l in labels])
    data = np.delete(arr, 0, axis=1)
    if binary_encoding:
        data[np.where(data != 0)] = 1
    else:
        data = normalize(data, 'max', axis=0)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=random_seed)
    _, filenames_test = train_test_split(filenames, test_size=0.25, random_state=random_seed)
    return x_train, x_test, y_train, y_test, filenames_test


# network used by geo et.al in the lemna paper. This is essentially the network from grosse et.al for the drebin dataset
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


def train_network(batch_size, epochs, loss, optimizer, vec_output, final_nonlinearity, random_seed):
    if not os.path.isdir('models'):
        os.makedirs('models')
    x_train, x_test, y_train, y_test, filenames_test = get_train_data_test_data(random_seed)
    # np.save('data_mimicus/test_data/float_encoded/test_data.npy', x_test)
    # np.save('data_mimicus/test_data/float_encoded/test_labels.npy', y_test)
    no_features = x_train.shape[1]
    model = get_network(no_features, final_nonlinearity, vec_output)
    fname = 'models/model.{epoch:03d}--ACC_{val_accuracy:.4f}--FP_{val_false_positive_metric:.4f}--' \
            'TP_{val_true_positive_metric:.4f}.hdf5'
    model_checkpoint_fp = ModelCheckpoint(fname, monitor='val_false_positive_metric', save_best_only=True, mode='min')
    model_checkpoint_tp = ModelCheckpoint(fname, monitor='val_true_positive_metric', save_best_only=True, mode='max')
    model.compile(optimizer, loss, metrics=['accuracy',custom_true_positive_metric(vec_output),
                                            custom_false_positive_metric(vec_output),], )
    print(model.summary())
    model.fit(x_train, y_train, batch_size, epochs, validation_data=(x_test, y_test), verbose=2,
              callbacks=[model_checkpoint_tp, model_checkpoint_fp])
    get_statistics(model, x_test, y_test)


# prints accuracy, precision, recall, fpr and f1 score for given model and test set with labels
def get_statistics(model, x_test, y_test):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    assert len(y_pred) == len(y_test)
    acc = np.sum(y_pred==y_test)/np.float(len(y_pred))
    cm = confusion_matrix(y_test, y_pred)
    TN, FN, TP, FP = cm[0,0], cm[1,0], cm[1,1], cm[0,1]
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    precision = TP/(TP+FP)
    F1 = 2*TP/(2*TP+FP+FN)
    print('The model achieved: Accuracy:{}, Precision:{}, Recall:{}, FPR:{}, F1 score:{} on the test set.'.format(
        acc, precision, TPR, FPR, F1))


if __name__ == '__main__':
    train_network(batch_size, epochs, loss, optimizer, vec_output, final_nonlinearity, random_seed)
