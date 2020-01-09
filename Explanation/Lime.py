import numpy as np
import pickle as pkl
import argparse
import time
import sys
from sklearn.linear_model import Ridge
from tqdm import tqdm
from scipy.spatial.distance import cosine
from lemna_postprocessing_scripts.relevances_to_linreg import linreg_relevances_to_vector_space
from scipy.sparse import load_npz


# returns l^2 weight for two points
def get_weight(x,y,sigma=1.0):
    dist = cosine(x,y)
    return np.exp(-dist/sigma)


# calculates lime weights for each feature
# perturbations is a 2d numpy array where the first row corresponds to the original sample
# labels is a 1d numpy array containing the labels for the perturbations and the original sample is supposed to be
# given label 1 by the classifier always. (this way, positive relevances will always speak _for_ the original
# classification of the classifier)
def get_lime_weights(perturbations, labels, random_state):
    assert perturbations.shape[0] == labels.shape[0]
    model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=random_state)
    weights = np.array([get_weight(perturbations[0], y) for y in perturbations])
    model_regressor.fit(perturbations, labels, sample_weight=weights)
    return model_regressor.coef_


# calculates lime weights for several perturbations and labels
# perturbations is a list where each list entry is a 2d numpy array suitable for get_lime_weights
# labels is a 2d numpy array with shape (no_samples, no_perturbations)
def explain_samples(perturbations, labels, random_state=None):
    relevances = []
    start_time = time.time()
    for p, l in tqdm(zip(perturbations, labels), total=len(perturbations)):
        w = get_lime_weights(p, l, random_state)
        relevances.append(w)
    end_time = time.time()
    print('Calculation of {} relevances took on {} seconds ({} seconds per sample).'.format(len(perturbations),
                                                                                           end_time-start_time,
                                                                                            (end_time-start_time)/
                                                                                            len(perturbations),
                                                                                           ))
    return relevances

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implementation of the LIME algorithm.')
    parser.add_argument('perturbation_path', type=str,
                        help='Path to list (.pkl) of perturbations for data of shape (no_perturbations, no_features).')
    parser.add_argument('label_path', type=str,
                        help='Path to array containing labels of perturbations of shape (no_samples, no_perturbations).'
                             'Labels are assumed to be binary (0/1).')
    parser.add_argument('save_path', type=str, help='Folder to save results.')
    parser.add_argument('--data_path', type=str, help='Path to data. Can be .npy, .npz, .pkl (sparse,numpy,list)')
    args = parser.parse_args()
    perturbations = pkl.load(open(args.perturbation_path, 'rb'))
    labels = np.load(args.label_path)
    if args.data_path:
        is_sparse = args.data_path.split('.')[-1] == 'npz'
        if args.data_path.split('.')[-1] == 'npy':
            data = np.load(args.data_path)
        elif args.data_path.split('.')[-1] == 'pkl':
            data = pkl.load(open(args.data_path, 'rb'))
        elif args.data_path.split('.')[-1] == 'npz':
            data = load_npz(args.data_path)
        else:
            print('Data format was not understood. Data could not be loaded.')
            sys.exit(1)
    rels = explain_samples(perturbations, labels)
    if args.data_path:
        linreg_relevances_to_vector_space(rels, data, args.save_path, is_sparse)
    else:
        pkl.dump(rels, open(args.save_path+'relevances_lime.pkl', 'wb'))
