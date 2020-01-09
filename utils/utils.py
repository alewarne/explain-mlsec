# some useful methods for this project like normalization of relevances for various datatypes
import numpy as np
import copy
import sys
import os
import pickle as pkl
from scipy import sparse
from tqdm import tqdm


# given relevance array (either numpy or csr_sparse) normalizes it with respect to given method
# method can be one of [mean, max, abs_max].
# if macro is False, each sample will be normalized for itself, else the whole data is normalized
def normalize_relevances(relevances, method, macro=False):
    if method not in ['mean', 'max', 'abs_max']:
        print('Invalid method name! Choose one of {}'.format(['mean', 'max', 'abs_max']))
        sys.exit(1)
    normed_relevances = copy.deepcopy(relevances)
    if macro:
        normed_relevances = to_macro(normed_relevances)
    if type(normed_relevances).__module__ == 'scipy.sparse.csr':
        nonzero_rows = normed_relevances.nonzero()[0]
        print('Calculating normalization for {} samples ...'.format(normed_relevances.shape[0]))
        for idx in tqdm(np.unique(nonzero_rows)):
            data_idx = np.where(nonzero_rows == idx)[0]
            normed_row = normalize_array(normed_relevances.data[data_idx], method)
            normed_relevances.data[data_idx] = normed_row
    elif type(normed_relevances).__module__ == 'numpy' or type(normed_relevances) is list:
        print('Calculating normalization for {} samples ...'.format(len(normed_relevances)))
        for i in range(len(normed_relevances)):
            if type(normed_relevances[i]) is list:
                normed_row = normalize_array(np.array(normed_relevances[i]), method)
                normed_relevances[i] = list(normed_row)
            else:
                normed_row = normalize_array(normed_relevances[i], method)
                normed_relevances[i] = normed_row
    else:
        print('Datatype not understood!')
        sys.exit(1)
    return normed_relevances


# normalizes 1D numpy array with respect to some method
def normalize_array(arr, method):
    if method not in ['mean', 'max', 'abs_max']:
        print('Invalid method name! Choose one of {}'.format(['mean', 'max', 'abs_max']))
        sys.exit(1)
    arr_cpy = copy.deepcopy(arr)
    if method == 'abs_max':
        abs_max = np.max(np.abs(arr_cpy))
        if abs_max != 0:
            arr_cpy = 1. / abs_max * arr_cpy
    elif method == 'mean':
        mu = np.mean(arr_cpy)
        sigma = np.std(arr_cpy)
        if sigma == 0:
            sigma += 1e-5
        arr_cpy = (arr_cpy-mu)/sigma
    elif method == 'max':
        min, max = np.min(arr_cpy), np.max(arr_cpy)
        if max != min:
            arr_cpy = (arr_cpy-min)/(max-min)
            arr_cpy = 2 * arr_cpy - 1
        else:
            arr_cpy = (arr_cpy - min) / min
    return arr_cpy


def to_macro(normed_relevances):
    if type(normed_relevances).__module__ == 'scipy.sparse.csr':
        macro = np.array([normed_relevances.data])
    elif type(normed_relevances).__module__ == 'numpy':
        macro = np.array([normed_relevances.flatten])
    elif type(normed_relevances) is list:
        macro = np.array([x for l in normed_relevances for x in l])
    return macro


def get_error_type(y_true, y_pred):
    if y_true == 0:
        if y_pred == 0:
            return 'TN'
        else:
            return 'FP'
    else:
        if y_pred == 0:
            return 'FN'
        else:
            return 'TP'


# takes a filepath and loads the contained data for the data formats .npy, .pkl, .npz
def load_npy_npz_pkl(path_to_data):
    _, filetype = os.path.splitext(path_to_data)
    if filetype == '.npz':
        data = sparse.load_npz(path_to_data)
    elif filetype == '.npy':
        data = np.load(path_to_data)
    elif filetype == '.pkl':
        data = pkl.load(open(path_to_data, 'rb'))
    else:
        print('Could not load filepath! Invalid datatype!')
        data = None
    return data
