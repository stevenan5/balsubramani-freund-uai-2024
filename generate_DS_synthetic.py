import os
import copy

from wrench.labelmodel import DawidSkene

import numpy as np
from scipy.io import savemat
from numpy.random import dirichlet
from numpy.random import beta
from numpy.random import multinomial

def generate_dataset(p, n, n_val):
    # p rules
    # n training datapoints
    # n_val validation points
    # assume 2 classes
    k = 2

    # dirichlet parameters for label distribution
    dir_param = np.ones(2)
    # label dist
    label_dist = dirichlet(dir_param, size=1).squeeze()

    # fix mean, set the variance ourself
    beta_mean = 0.6
    beta_a = 2
    beta_b = (1 / beta_mean - 1) * beta_a

    # generate confusion matrices for all rules
    # do one coin dawid skene
    cm = np.zeros((p, k, k))
    for j in range(p):
        cm[j, 0, 0] = beta(beta_a, beta_b)
        # cm[j, 1, 1] = beta(beta_a, beta_b)
        cm[j, 1, 1] = cm[j, 0, 0]
        cm[j, 0, 1] = 1 - cm[j, 0, 0]
        cm[j, 1, 0] = 1 - cm[j, 1, 1]

    # generate eta
    val_labels = np.zeros(n_val)
    train_labels = np.zeros(n)
    for i in range(n_val):
        val_labels[i] = multinomial_argmax(label_dist)
    for i in range(n):
        train_labels[i] = multinomial_argmax(label_dist)

    # generate observed ground truth
    val_labels = val_labels.astype(int)
    train_labels = train_labels.astype(int)

    # generate predictions
    train_pred = np.zeros((n, p))
    val_pred = np.zeros((n_val, p))

    for j in range(p):
        for i in range(n_val):
            val_pred[i, j] = multinomial_argmax(cm[j, val_labels[i], :])

        for i in range(n):
            train_pred[i, j] = multinomial_argmax(cm[j, train_labels[i], :])

    val_pred = val_pred.astype(int)
    train_pred = train_pred.astype(int)

    # get the DS preds, which are the underlying distribution eta
    lm = DawidSkene()
    lm.model_type = 'one_coin'
    ds_pred = lm.predict_proba([train_pred, train_labels], oracle=True)

    mdic = {
            "train_labels": train_labels,
            "train_pred": train_pred,
            "validation_labels": val_labels,
            "val_pred": val_pred,
            "ds_pred": ds_pred,
            }

    return mdic

def multinomial_argmax(dist):
    return np.argmax(multinomial(1, dist))

if __name__ == '__main__':

    # number of datasets to make
    n_datasets = 10
    # n_datasets = 1

    # choose number of rules
    n_rules = 3

    # choose number of datapoints
    n_train = [100, 1000, 10000, 100000]

    # choose number of validation points
    n_valid = 100

    if not os.path.exists('./datasets/synthetic_dawid_skene/'):
        os.makedirs('./datasets/synthetic_dawid_skene/')


    for o in range(n_datasets):
        dataset = generate_dataset(n_rules, n_train[-1], n_valid)
        for n_t in n_train:
            path = './datasets/synthetic_dawid_skene/synth_' + str(n_rules) + \
                    'p_' + str(n_t) +\
                    'n_' + str(n_valid) + 'nval__'
            full_path = path + str(o) + '.mat'
            dataset_tmp = copy.deepcopy(dataset)
            dataset_tmp['train_pred'] = dataset_tmp['train_pred'][:n_t, :]
            dataset_tmp['train_labels'] = dataset_tmp['train_labels'][:n_t]
            dataset_tmp['ds_pred'] = dataset_tmp['ds_pred'][:n_t, :]
            savemat(full_path, dataset_tmp)
