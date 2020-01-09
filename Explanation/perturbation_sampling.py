import numpy as np
from scipy import sparse
import pickle as pkl
import sys
import os
import argparse
from tqdm import tqdm
from keras.models import load_model

sys.path.append('../utils/')
from utils import load_npy_npz_pkl


# samples perturbations from a data_sample (vector) by choosing a random number of random features from the original
# sample and setting the non-chosen features to 0
def sample_data_points(data_sample, no_samples):
    if type(data_sample).__module__ == 'scipy.sparse.csr':
        is_sparse = True
        # in sparse case we remember all row and column indices of the perturbations and create the matrix at one point
        nonzero_row_indices, nonzero_column_indices = [], []
    else:
        is_sparse = False
        # in the non-sparse case we save the perturbation directly after creation
        samples = np.zeros(shape=(no_samples,) + data_sample.shape, dtype=data_sample.dtype)
    non_zeros = np.nonzero(data_sample)
    # if data_sample is a (sparse) vector each nonzero index appears exactly once in the non_zeros
    if is_sparse:
        no_nonzero_entries = len(non_zeros[0])
        sampling_values = non_zeros[1]
    elif len(data_sample.shape) < 2:
        no_nonzero_entries = len(non_zeros[0])
        sampling_values = non_zeros[0]
    # if data sample is a vector of vectors, each nonzero index appears multiple times for each vector
    else:
        no_nonzero_entries = len(np.unique(non_zeros[0]))
        sampling_values = np.unique(non_zeros[0])
    for i in range(no_samples):
        # the first row contains the original sample
        if i == 0:
            no_samples_indices = no_nonzero_entries
        else:
            # how many entries are we going to draw (at least one!)
            no_samples_indices = np.random.randint(1, no_nonzero_entries + 1)
        # which samples are we actually drawing
        if is_sparse:
            sample_indices = np.random.choice(non_zeros[1], no_samples_indices, replace=False)
            nonzero_row_indices += [i] * no_samples_indices
            nonzero_column_indices += list(sample_indices)
            continue
        else:
            sample_indices = np.random.choice(sampling_values, no_samples_indices, replace=False)
            perturbed_data = np.zeros(shape=data_sample.shape)
            perturbed_data[sample_indices] = data_sample[sample_indices]
            samples[i][:] = perturbed_data
    if is_sparse:
        data = [1] * len(nonzero_row_indices)
        samples = sparse.csr_matrix((data, (nonzero_row_indices, nonzero_column_indices)),
                                    shape=(no_samples, data_sample.shape[1]), dtype=data_sample.dtype)
    return samples


# given  this method creates 'no_samples' perturbations of each datapoint in data. Careful, perturbations can easily
# become very big in memory. Pre calculate if no_samples*data fits into memory.
def get_pertubations(data, no_samples):
    perturbations = []
    if type(data).__module__ in ['scipy.sparse.csr', 'numpy']:
        total = data.shape[0]
    else:
        total = len(data)
    print('Sampling data points...')
    for data_sample in tqdm(data, total=total):
        perturbations.append(sample_data_points(data_sample, no_samples))
    # if all data points have the same shape, save one big numpy array
    if type(data).__module__ == 'scipy.sparse.csr':
        perturbations = sparse.vstack(perturbations)
    return perturbations


# classifies a batch of perturbation data. We assume that samples are of shape (no_perturbations, sample_dimension)
# and that the model can predict this sort of data
def get_classification(model, samples, batch_size=500):
    # if samples is list we assume that each sample has a different shape and we have to classify sample-wise
    if type(samples) is list:
        labels = []
        for sample in samples:
            labels.append(np.argmax(model.predict(sample.reshape((1,)+sample.shape)), axis=1))
        labels = np.array(labels).reshape((len(samples),))
    # else we can just predict the entire sample set
    else:
        labels = np.argmax(model.predict(samples, batch_size=batch_size), axis=1)
    labels = labels.astype(np.uint8)
    # we take care that label 1 is the label the classifier assigns to the sample (which is in row 0 of the
    # perturbations) and 0 is the one of differently classified perturbations
    classifier_label = labels[0]
    targets = np.where(labels == classifier_label)
    nontargets = np.where(labels != classifier_label)
    labels[targets] = 1
    labels[nontargets] = 0
    return labels


# transforms a tuple of non zero indices (like output of scipy.sparse.nonzero or np.nonzero) to suited representation
# for linear regression. Assumes original sample in the first row
def perturbation_block_to_regression_sample(nonzero_tuple):
    nonzero_indices_rows, nonzero_indices_columns = nonzero_tuple[0], nonzero_tuple[1]
    nonzero_samples_indices = nonzero_indices_columns[np.where(nonzero_indices_rows==0)]
    feature_size = len(np.unique(nonzero_samples_indices))
    no_samples = len(np.unique(nonzero_indices_rows))
    orig_idx_2_reg_idx = dict(zip(np.unique(nonzero_samples_indices), range(feature_size)))
    linreg_block = np.zeros(shape=(no_samples, feature_size), dtype=np.uint8)
    for row_no in np.unique(nonzero_indices_rows):
        nonzero_entries = nonzero_indices_columns[np.where(nonzero_indices_rows == row_no)[0]]
        reg_indices = [orig_idx_2_reg_idx[idx] for idx in np.unique(nonzero_entries)]
        linreg_block[row_no, reg_indices] = 1
    return linreg_block


# takes (test) data and computes perturbations and the labels of the perturbations aswell as a linear representation of
# the perturbation data. Delete specifies if the non-selected features of the perturbations will be deleted
# or (if false) set to zero.
def perturbation_pipeline(data, model, no_perturbations_per_sample, save_path, save_perturbations, delete):
    seed = 40
    np.random.seed(seed)
    no_samples = data.shape[0] if not type(data) is list else len(data)
    all_labels, all_linregs, all_perturbations = [], [], []
    print('Computing perturbations for {} samples ...'.format(no_samples))
    for data_sample in tqdm(data):
        if len(data_sample.shape) < 3 and delete:
            print('Error. Delete = 1 is only allowed for sequential data!')
            sys.exit(1)
        perturbations = sample_data_points(data_sample, no_perturbations_per_sample)
        if delete:
            perturbations_deleted = []
            for i in range(perturbations.shape[0]):
                zero_vectors = np.array([(x == 0).all() for x in perturbations[i]])
                indices_to_delete = np.where(zero_vectors > 0)[0]
                perturbations_deleted.append(np.delete(perturbations[i], indices_to_delete, axis=0))
            labels = get_classification(model, perturbations_deleted)
        else:
            labels = get_classification(model, perturbations)
        all_labels.append(labels)
        all_linregs.append(perturbation_block_to_regression_sample(perturbations.nonzero()))
        if save_perturbations:
            all_perturbations.append(perturbations)
    np.save(os.path.join(save_path, 'perturbation_labels_seed_{}.npy'.format(seed)), np.array(all_labels))
    pkl.dump(all_linregs, open(os.path.join(save_path, 'linreg_representations_seed_{}.pkl'.format(seed)), 'wb'))
    if save_perturbations:
        if type(data) is list:
            pkl.dump(all_perturbations, open(os.path.join(save_path, 'perturbation_data_seed_{}.pkl'.format(seed)), 'wb'))
        elif type(data).__module__ == 'scipy.sparse.csr':
            sparse.save_npz(os.path.join(save_path, 'perturbation_data_seed_{}.npz'.format(seed)), sparse.vstack(all_perturbations))
        else:
            np.save(os.path.join(save_path, 'perturbation_data_seed_{}.npy'.format(seed)), np.array(all_perturbations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perturbation sampling process for lemna algorithm.')
    parser.add_argument('data_path', type=str, help='Path to data structure containing data samples.')
    parser.add_argument('model_path', type=str, help='Path to a keras model (*.hdf5) that can be loaded with model.load().')
    parser.add_argument('save_path', type=str, help='Where to store the results.')
    parser.add_argument('no_perturbations', type=int, help='How many perturbations of each sample will be created.')
    parser.add_argument('--save_perturbations', type=int, default=0, help='If 1 the real perturbations (not only the'
                                                    'binary representation of them) will be saved. This can be useful'
                                                    'for debugging but can use a lot of memory.')
    parser.add_argument('--delete', type=int, default=0, help='If 1, features that are not selected for a perturbation'
                                                              'will be deleted from the sample instead of setting them'
                                                              'to zero.')
    args = vars(parser.parse_args())
    for k,v in args.items():
        print('{} = {}'.format(k, v))
    data = load_npy_npz_pkl(args['data_path'])
    model = load_model(args['model_path'])
    perturbation_pipeline(data, model, args['no_perturbations'], args['save_path'], bool(args['save_perturbations']),
                          bool(args['delete']))

