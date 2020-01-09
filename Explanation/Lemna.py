import numpy as np
import cvxpy as cvx
from sklearn import linear_model
import argparse
import multiprocessing
import pickle as pkl
import time
import sys


# gaussian density function
def gaussian(x, mu, sigma_squared):
    eps = 0
    return 1/(np.sqrt(2*np.pi*sigma_squared)+eps)*np.exp(-0.5*(x-mu)**2/(sigma_squared+eps))


# the expectation maximization algorithm of the lemna paper, calculation of indices can be found in appendix
def em_regression_algorithm(data, labels, K, alpha_S, iterations, linreg_type, verbose=True, save_path=None):
    # determine if data is sparse
    sparse = True if type(data).__module__ == 'scipy.sparse.csr' else False
    no_samples = data.shape[0]
    sample_len = data.shape[1]
    label_sum = np.sum(np.abs(labels))
    no_ones = len(np.where(labels == 1)[0])
    no_zeros = len(labels) - no_ones
    #data = (data-np.mean(data))/np.std(data)
    if linreg_type not in ['lasso', 'fused_lasso']:
        print('Invalid linreg_type (%s)' %linreg_type)
        exit(1)
    elif sample_len <=1:
        if verbose:
            print('Encountered invaliddata sample!')
        with open(save_path, 'a') as f:
            print('Invalid sample.', file=f)
        return np.array([-1]), np.array([-1])
    eps = 1e-6
    number_of_history_betas = 3
    convergence_threshold = 1e-2
    # initialize the parameters randomly
    pi, sigma_sq = np.random.uniform(0, 1, size=K), np.random.uniform(0, 1, size=K)
    # normalize pi
    pi = 1/np.sum(pi) * pi
    beta = np.random.uniform(-.1, .1, size=(K, sample_len))
    z_hat = np.zeros(shape=(no_samples, K))
    # check for convergence using last betas
    old_likelihoods = []
    converged = False
    # run at most 'iterations' iterations but finish if the last 'number of history betas' log likelihood values are
    # close to each other
    initial_log_likelihood = 0
    for n in range(no_samples):
        if sparse:
            likelihood = sum([pi[k] * gaussian(labels[n], data.getrow(n).dot(beta[k,:])[0], sigma_sq[k])
                              for k in range(K)])
        else:
            likelihood = sum([pi[k] * gaussian(labels[n], np.dot(data[n,:],beta[k, :]), sigma_sq[k])
                              for k in range(K)])
        if likelihood != 0:
            initial_log_likelihood += np.log(likelihood)
    if verbose:
        print('Starting Expectation maximization algorithm for %d iterations with sum of labels %d' %(iterations,
                                                                                                      label_sum))
    start_time = time.time()
    for iter in range(iterations):
        # E step
        for i in range(no_samples):
            if sparse:
                denom_e = sum([pi[k] * gaussian(labels[i], data.getrow(i).dot(beta[k, :])[0], sigma_sq[k]) for k in
                             range(K)])
            else:
                denom_e = sum([pi[k] * gaussian(labels[i], np.dot(data[i,:], beta[k,:]), sigma_sq[k]) for k in
                             range(K)])
            if denom_e == 0:
                denom_e = eps
                if verbose:
                    print('set denom_e to eps')
            for k in range(K):
                pred_2 = data.getrow(i).dot(beta[k, :])[0] if sparse else np.dot(data[i,:], beta[k,:])
                z_hat[i, k] = pi[k]*gaussian(labels[i], pred_2, sigma_sq[k])/denom_e
        # M step
        for k in range(K):
            denom_m = np.sum(z_hat[:, k])
            if denom_m == 0:
                denom_m = eps
            if sparse:
                sigma_sq[k] = sum([z_hat[i, k] * (labels[i] - data.getrow(i).dot(beta[k, :])[0])**2 for i in
                                   range(no_samples)]) / denom_m
            else:
                sigma_sq[k] = sum([z_hat[i, k] * (labels[i] - np.dot(data[i, :], beta[k, :])) ** 2 for i in
                                   range(no_samples)]) / denom_m
            if sigma_sq[k] == 0:
                sigma_sq[k] += eps
                if verbose:
                    print('added eps to sigma')
            pi[k] = np.sum(z_hat[:, k])/no_samples
        component_assignments = np.argmax(z_hat, axis=1)
        # estimate betas by linear regression with fused lasso loss
        for k in range(K):
            sample_indices_of_k = np.where(component_assignments == k)[0]
            samples_of_k = data[sample_indices_of_k, :]
            labels_of_k = labels[sample_indices_of_k]
            if len(labels_of_k) > 0:
                if linreg_type == 'fused_lasso':
                    beta[k,:] = solve_fused_lasso_regression(samples_of_k, labels_of_k, alpha_S)
                elif linreg_type == 'lasso':
                    reg = linear_model.Lasso(alpha=alpha_S, precompute=True, normalize=True, max_iter=3000)
                    reg.fit(samples_of_k, labels_of_k)
                    beta[k, :] = reg.coef_
        # recompute log_likelihood in order to check for convergence
        log_likelihood = 0
        for n in range(no_samples):
            if sparse:
                likelihood = sum([pi[k] * gaussian(labels[n], data.getrow(n).dot(beta[k, :])[0], sigma_sq[k])
                                  for k in range(K)])
            else:
                likelihood = sum([pi[k] * gaussian(labels[n], np.dot(data[n, :], beta[k, :]), sigma_sq[k])
                                  for k in range(K)])
            if likelihood != 0:
                log_likelihood += np.log(likelihood)
        if len(old_likelihoods) < number_of_history_betas:
            old_likelihoods.append(log_likelihood)
        else:
            abs_diffs = []
            for beta_idx in range(number_of_history_betas):
                diff = np.abs(old_likelihoods[beta_idx]-log_likelihood)
                abs_diffs.append(diff)
            convergence_check = [np.sum(diff <= convergence_threshold) for diff in abs_diffs]
            if np.sum(convergence_check) == number_of_history_betas:
                converged = True
            old_likelihoods.pop(0)
            old_likelihoods.append(log_likelihood)
        if verbose:
            print('likelihood history', old_likelihoods)
        if converged:
            end_time = time.time()
            if verbose:
                print('EM-Alogirthm converged after %d iterations (%d seconds).' %(iter, end_time-start_time))
                argm = np.argmax(z_hat, axis=1)
                for k in range(K):
                    indices_of_k = np.where(argm==k)[0]
                    labels_of_k = labels[indices_of_k]
                    labels_in_k = np.unique(labels_of_k)
                    d = {}
                    for label in labels_in_k:
                        d[label] = len(np.where(labels_of_k==label)[0])
                    print('labels in cluster %d'%k, d)
            break
    if save_path:
        with open(save_path, 'a') as f:
            projections = np.dot((beta * pi[:, np.newaxis]), np.transpose(data))
            projections = np.sum(projections, axis=0)
            diff = (projections - labels) ** 2
            mse = 1. / len(diff) * np.sum(diff)
            if converged:
                print('S=%.3f_K=%d_linreg_type=%s_no_ones=%d_no_zeros=%d_time=%.4f_mse=%.4f'
                      % (alpha_S, K, linreg_type, no_ones, no_zeros, end_time - start_time, mse), file=f)
            else:
                print('S=%.3f_K=%d_linreg_type=%s_no_ones=%d_no_zeros=%d_time=%.4f_mse=%.4f'
                      % (alpha_S, K, linreg_type, no_ones, no_zeros, -1, mse), file=f)
            # print('mse', mse)
    # return the parameters by choosing the cluster belonging to the first row of the perturbations which is by
    # assumption the sample to be explained
    cluster_idx_sample = np.argmax(z_hat[0])
    return beta[cluster_idx_sample], sigma_sq[cluster_idx_sample]


# returns matrix A such that sum(abs(A*x)) is the fused lasso constraint on x
def get_band_matrix_fused_lasso(dim):
    if dim <= 1:
        print('Invalid dimension for band matrix (%d)!'%dim)
        return None
    A = np.diag(-1*np.ones(dim))
    rng = np.arange(dim-1)
    A[rng, rng+1] = 1
    A[dim-1,:] = 0
    return A


def solve_fused_lasso_regression(samples, labels, S):
    # for the sake of clarity
    A = cvx.Constant(samples)
    no_dimensions = samples.shape[1]
    beta = cvx.Variable(no_dimensions)
    # careful: the band matrix can get large very fast if dimension is high
    # D = get_band_matrix_fused_lasso(no_dimensions)
    regularization = beta[1:] - beta[:no_dimensions - 1]
    objective = cvx.Minimize(cvx.sum_squares(A*beta - labels))
    # the constraint is the sum of the (absolute) differences of the neighbored betas to be bounded by S
    # constraints = [cvx.sum(cvx.abs(D*beta)) <= S]
    constraints = [cvx.sum(cvx.abs(regularization)) <= S]
    problem = cvx.Problem(objective, constraints)
    problem.solve()
    return beta.value


def lemna_parallel(perturbation_data, perturbation_labels, K, alpha_S, iterations, no_processes, linreg_type,
                   repetitions=1, verbose=False, save_path=None):
    assert len(perturbation_data) == len(perturbation_labels)
    no_samples = len(perturbation_data) * repetitions
    # repeat each perturbation repetitions times for parallel processing
    perturbations_repeated = []
    for p in perturbation_data:
        perturbations_repeated += [p]*repetitions
    labels_repeated = np.repeat(perturbation_labels, repetitions, axis=0)
    if save_path:
        filenames = np.array([save_path+str(i) for i in range(len(perturbation_data))])
        filenames = np.repeat(filenames, repetitions)
    else:
        filenames = no_samples*[None]
    arg_gen = zip(perturbations_repeated, labels_repeated, no_samples*[K], no_samples*[alpha_S], no_samples*[iterations],
                  no_samples*[linreg_type], no_samples*[verbose], filenames)
    with multiprocessing.Pool(processes=no_processes) as pool:
        lemna_betas = pool.starmap(em_regression_algorithm, arg_gen)
    if type(perturbation_data) is list:
        betas = [lemna_beta[0] for lemna_beta in lemna_betas]
    else:
        betas = np.array([lemna_beta[0] for lemna_beta in lemna_betas]).reshape((len(perturbation_data), repetitions,
                                                                                 perturbation_data.shape[-1]))
    sigmas = np.array([lemna_beta[1] for lemna_beta in lemna_betas]).reshape((len(perturbation_data), repetitions))
    return betas, sigmas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implementation of the Lemna algorithm.')
    parser.add_argument('data_path', type=str,
                        help='Path to list (.pkl) containing perturbations of shape (no_perturbations, no_features).')
    parser.add_argument('label_path', type=str,
                        help='Path to array containing labels of shape (no_samples, no_perturbations). Labels are'
                             ' assumed to be binary (0/1).')
    parser.add_argument('save_path', type=str, help='Runtime, mse and more is documented in a file for each sample.')
    parser.add_argument('K', type=int, help='K parameter (number of components) of the algorithm.')
    parser.add_argument('linreg_type', nargs=2,
                        help='Lasso for linear regression with L1 regularization, fused_lasso for fused lasso. Second'
                             'parameter is alpha for lasso and S for fused_lasso.')
    parser.add_argument('iterations', type=int, help='Number of maximum iterations during EM Algorithm.')
    parser.add_argument('repetitions', type=int, help='Number of repetitions of EM Algorithm as its'
                                                                       'output is not deterministic.')
    parser.add_argument('--no_processes', type=int, default=2, help='Number of processes for running parallel.')
    parser.add_argument('--verbose', type=int, default=0, help='Detailed output of EM algorithm.')
    args = vars(parser.parse_args())
    for k,v in args.items():
        print('{} = {}'.format(k, v))
    if args['data_path'].split('.')[-1] == 'npy':
        data = np.load(args['data_path'])
    elif args['data_path'].split('.')[-1] == 'pkl':
        data = pkl.load(open(args['data_path'], 'rb'))
    else:
        print('Data format was not understood. Data could not be loaded.')
        sys.exit(1)
    labels = np.load(args['label_path'])
    betas, sigmas = lemna_parallel(data, labels, args['K'], float(args['linreg_type'][1]), args['iterations'],
                                       args['no_processes'], args['linreg_type'][0], repetitions=args['repetitions'],
                                       verbose=bool(args['verbose']), save_path=args['save_path']
                                       )
    if type(betas) is list:
        pkl.dump(betas, open(args['save_path'] + 'K=%d_S=%.4f_betas.pkl'%(args['K'], float(args['linreg_type'][1])), 'wb'))
    else:
        np.save(args['save_path'] + 'K=%d_S=%.4f_betas.npy'%(args['K'], float(args['linreg_type'][1])), betas)
    np.save(args['save_path'] + 'K=%d_S=%.4f_sigmas.npy' % (args['K'], float(args['linreg_type'][1])), sigmas)
