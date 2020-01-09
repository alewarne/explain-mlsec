# Explanations

This folder contains scripts to create explanations for the network architectures using LIME ([Ribeiro et al.](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)) and LEMNA ([Guo et al.](http://people.cs.vt.edu/gangwang/ccs18.pdf)). Based on the finding of our paper we do not recommend to use these methods but [white-box methods](https://github.com/albermax/innvestigate) instead. Still, we want to publish our implementations for the sake of open access.

* The input data for usage of this repository can be of three types: numpy array of shape (n_samples, n_features) like in Mimicus or VulDeePecker, scipy.sparse.csr_matrix of shape (n_samples, n_features) like in Drebin or a list of length n_samples where each entry in the list is a numpy array of different length like in DAMD, for example.
* To use LIME or LEMNA you firstly need perturbations of the data. You can call  `python3 perturbation_sampling.py --help` to find out how to generate them.
* With the perturbations you can calculate relevances for the features your models use. Check `python3 Lemna.py --help` or `python3 Lime.py --help` to find out how.