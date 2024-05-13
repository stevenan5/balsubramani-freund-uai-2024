import os
import json
import logging
from time import perf_counter

import numpy as np
import scipy as sp
from numpy.matlib import repmat
from scipy.io import savemat
import scipy.io as sio
import matplotlib.pyplot as plt

from wrench._logging import LoggingHandler
from wrench.labelmodel import WMRC
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def run_wmrc_ds_comp(
        dataset_prefix,
        dataset_name=None,
        n_classes=0,
        constraint_form='confusion_matrix',
        add_mv_const=False,
        labeled_set='train',
        bound_method='binomial',
        use_inequality_consts=False,
        n_max_labeled = -1,
        n_runs = 1,
        verbose=False,
        # verbose=True,
        save_path=None,
        solver='MOSEK',
        logger=logger,
        ):

    #### Load dataset
    dataset_path = os.path.join(dataset_prefix, dataset_name + '.mat')
    data = sio.loadmat(dataset_path)
    train_data = [data['train_pred'], data['train_labels']]
    ds_pred = data['ds_pred']
    n_train_points = train_data[0].shape[0]
    n_train_rules = train_data[0].shape[1]

    # result path
    # only add datapoint count if we're not oracle
    if n_max_labeled > 0:
        n_labeled = n_max_labeled
    else:
        n_labeled = None
    result_filename = get_result_filename(dataset, constraint_form,
            labeled_set, bound_method, use_inequality_consts, add_mv_const,
            n_labeled=n_labeled)

    use_all_valid = False
    try:
        valid_data = [data['val_pred'], data['validation_labels']]
        use_all_valid = labeled_set == 'valid' and\
                n_max_labeled == valid_data[0].shape[0]
    except KeyError:
        print('No validation set found!')


    if labeled_set == 'valid':
        labeled_data = valid_data
    else:
        labeled_data = train_data

    ### if in oracle setting, always force n_runs to be 1
    # or if we're using all validation points or replotting
    is_oracle = labeled_set == 'train' and not use_inequality_consts
    if is_oracle or use_all_valid or bound_method == 'unsupervised':
        n_runs = 1

    #### Run label model: WMRC
    label_model = WMRC(solver=solver, verbose=verbose)

    for run_no in range(n_runs):
        if n_runs > 1:
            logger.info('------------Run Number %d------------', run_no + 1)
        n_fit_runs = 0
        # reset problem status so we can refit
        label_model.prob_status = None
        start_time = perf_counter()
        label_model.fit(
                train_data,
                labeled_data,
                constraint_type=constraint_form,
                bound_method=bound_method,
                use_inequality_consts=False,
                n_max_labeled=n_max_labeled,
                logger=logger,
                )
        end_time = perf_counter()

        elapsed_time = end_time - start_time

        ### make predictions
        Y_p_train = label_model.predict_proba(train_data)
        # pred_train = np.argmax(Y_p_train, axis=1)
        true_labels_train = np.squeeze(train_data[1])

        ### compute losses
        brier_score_train = multi_brier(true_labels_train, Y_p_train)
        logloss_train = log_loss(true_labels_train,Y_p_train, labels=[0, 1])
        acc_train = label_model.test(train_data, 'acc')

        # compute kl divergence of DS pred to BF pred
        entropy = np.sum(sp.stats.entropy(ds_pred, axis=1))
        kl_from_ds = -1 * np.sum(ds_pred * np.log(Y_p_train)) - entropy
        kl_from_ds /= n_train_points

        # only run calibration code if n_classes is 2
        if n_classes == 2:
            prob_true_train, prob_pred_train = calibration_curve(
                    np.squeeze(train_data[1]),
                    np.clip(Y_p_train[:, 1], 0, 1),
                    n_bins=10)

        if run_no == 0:
            mdic = {
                    "pred_train": [],
                    "log_loss_train": [],
                    "true_labels_train": true_labels_train,
                    "brier_score_train": [],
                    "acc_train": [],
                    "err_train": [],
                    "n_rules_used": [],
                    "rule_weights": [],
                    "class_freq_weights": [],
                    "avg_num_labeled_per_rule": [],
                    "fit_elapsed_time": [],
                    "wmrc_xent_ub": [],
                    "kl_from_ds": [],
                    }
            if n_classes == 2:
                mdic["x_calibration_train"] = []
                mdic["y_calibration_train"] = []

        mdic["pred_train"].append(Y_p_train)
        mdic["log_loss_train"].append(logloss_train)
        mdic["brier_score_train"].append(brier_score_train)
        mdic["acc_train"].append(acc_train)
        mdic["err_train"].append(1 - acc_train)
        mdic["kl_from_ds"].append(kl_from_ds)
        mdic["n_rules_used"].append(label_model.n_rules_used)
        mdic["rule_weights"].append(label_model.param_wts)
        mdic["class_freq_weights"].append(label_model.class_freq_wts)
        mdic["avg_num_labeled_per_rule"].append(\
                label_model.avg_labeled_per_rule)
        mdic["fit_elapsed_time"].append(elapsed_time)
        # optimization problem finds the negative entropy and is not
        # averaged over the total number of datapoints
        mdic["wmrc_xent_ub"].append(-1 * label_model.prob.value\
                / n_train_points)
        if n_classes == 2:
            # this is kind of screwy in terms of the resulting format
            mdic["x_calibration_train"].append(prob_pred_train)
            mdic["y_calibration_train"].append(prob_true_train)

        # if we're in the unsupervised setting and the bound scale changed,
        # record that. note the only time bound_scale changes is when
        # unsupervised bounds are used.

        ### report results
        logger.info('================Results================')
        logger.info('time to fit: %.1f seconds', elapsed_time)
        logger.info('train acc (train err): %.4f (%.4f)',
                acc_train, 1 - acc_train)
        logger.info('wmrc train log loss upper bound %.4f',
                mdic['wmrc_xent_ub'][-1])
        logger.info('train log loss: %.4f', logloss_train)
        logger.info('train brier score: %.4f', brier_score_train)
        logger.info('KL from DS %.4f', kl_from_ds)

    # if number of runs is >1, report and store mean results and standard
    # deviations
    if n_runs > 1:
        mdic["log_loss_train_mean"]     = np.mean(mdic["log_loss_train"])
        mdic["brier_score_train_mean"]  = np.mean(mdic["brier_score_train"])
        mdic["acc_train_mean"]          = np.mean(mdic["acc_train"])
        mdic["err_train_mean"]          = np.mean(mdic["err_train"])
        mdic["n_rules_used_mean"]       = np.mean(mdic["n_rules_used"])
        mdic["avg_num_labeled_per_rule_mean"] =\
                np.mean(mdic["avg_num_labeled_per_rule"])
        mdic["fit_elapsed_time_mean"]   = np.mean(mdic["fit_elapsed_time"])
        mdic["wmrc_xent_ub_mean"]       = np.mean(mdic["wmrc_xent_ub"])

        mdic["log_loss_train_std"]     = np.std(mdic["log_loss_train"])
        mdic["brier_score_train_std"]  = np.std(mdic["brier_score_train"])
        mdic["acc_train_std"]          = np.std(mdic["acc_train"])
        mdic["err_train_std"]          = np.std(mdic["err_train"])
        mdic["avg_num_labeled_per_rule_std"] =\
                np.std(mdic["avg_num_labeled_per_rule"])
        mdic["fit_elapsed_time_std"]   = np.std(mdic["fit_elapsed_time"])
        mdic["wmrc_xent_ub_std"]       = np.std(mdic["wmrc_xent_ub"])

        logger.info('================Aggregated Results================')
        logger.info('Total number of runs: %d', n_runs)
        logger.info('Average time to fit: %.1f seconds (std: %.1f)',
                mdic['fit_elapsed_time_mean'], mdic['fit_elapsed_time_std'])
        logger.info('Average of %.2f rules out of %d had labeled data',
                mdic["n_rules_used_mean"], n_train_rules)
        logger.info('Average of %d labeled datapoints out of %d per rule (std: %.4f)',
                mdic["avg_num_labeled_per_rule_mean"], n_max_labeled,\
                        mdic["avg_num_labeled_per_rule_std"])
        logger.info('train mean acc +- std (mean train err):'
                ' %.4f +- %.4f (%.4f)', mdic['acc_train_mean'],
                mdic['acc_train_std'], mdic['err_train_mean'])
        logger.info('wmrc train log loss upper bound mean +- std: %.4f +- %.4f',
                mdic['wmrc_xent_ub_mean'], mdic['wmrc_xent_ub_std'])
        logger.info('train mean log loss +- std: %.4f +- %.4f',
                mdic['log_loss_train_mean'], mdic['log_loss_train_std'])
        logger.info('train mean brier score +- std: %.4f +- %.4f',
                mdic['brier_score_train_mean'], mdic['brier_score_train_std'])

    savemat(os.path.join(save_path, result_filename), mdic)

    return mdic

def get_result_filename(dataset_name, constraint_name,
        labeled_set, bound_method, use_inequality_consts, add_mv_const,
        n_labeled=None):

    bound_name = ''
    if bound_method == 'unsupervised':
        est_pdgm = 'unsup_'
    else:
        est_pdgm = 'semisup_'
        if labeled_set == 'valid':
            bound_name = '_' + bound_method

    if use_inequality_consts:
        use_ineq = 'ineqconst'
    else:
        use_ineq = 'eqconst'

    if labeled_set == 'valid':
        labeled_set_used = 'validlabels_'
    else:
        labeled_set_used = 'trainlabels_'

    if add_mv_const:
        use_mv = '_add_mv'
    else:
        use_mv = ''

    n_lab = ''
    if n_labeled is not None:
        if n_labeled > 0 and use_inequality_consts and labeled_set == 'valid':
            n_lab = str(n_labeled)

    filename = "WMRC_"\
            + dataset_name + '_'\
            + constraint_name + '_'\
            + est_pdgm\
            + n_lab\
            + labeled_set_used\
            + use_ineq\
            + bound_name\
            + use_mv\
            + ".mat"

    return filename

def multi_brier(labels, pred_probs):
    """
    multiclass brier score
    assumes labels is 1d vector with elements in {0, 1, ..., n_class - 1}
    position of the ture class and 0 otherwise
    """
    n_class = int(np.max(labels) + 1)
    labels = (np.arange(n_class) == labels[..., None]).astype(int)
    sq_diff = np.square(labels - pred_probs)
    datapoint_loss = np.sum(sq_diff, axis=1)

    return np.mean(datapoint_loss)

# pylint: disable=C0103
if __name__ == '__main__':
    # we only want to test certain combinations
    # semi-supervised, validation set, use bounds
    # unsupervised, training set, use bounds (akin to crowdsourcing)
    # oracle, training set, no bounds (equality constraints)

    # create results folder if it doesn't exist
    results_folder_path = './results/synthetic_dawid_skene'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    dataset_prefix = './datasets/'

    datasets = []
    n_pts = [100, 1000, 10000, 100000]
    # change dataset path if using synthetic datasets
    dataset_prefix = os.path.join(dataset_prefix, 'synthetic_dawid_skene')
    # create dataset names for synthetic datasets
    synth_filename_part = lambda x: 'synth_3p_' + str(x) + 'n_100nval__'
    n_synth = 10
    for i in range(n_synth):
        for n_pt in n_pts:
            datasets.append(synth_filename_part(n_pt) + str(i))

    constraint_type = 'accuracy'
    # just need name and class number

    for dataset in datasets:

        # make result folder if it doesn't exist
        dataset_result_path = os.path.join(results_folder_path, dataset)
        if not os.path.exists(dataset_result_path):
            os.makedirs(dataset_result_path)
        # make folder for WMRC specifically
        method_result_path = os.path.join(dataset_result_path, 'WMRC')
        if not os.path.exists(method_result_path):
            os.makedirs(method_result_path)

        n_class = 2

        cons_result_path = os.path.join(
                method_result_path, constraint_type)

        if not os.path.exists(cons_result_path):
            os.makedirs(cons_result_path)

        # change loggers every time we change settings
        # remove old handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        formatter=logging.Formatter('%(asctime)s - %(message)s',
                '%Y-%m-%d %H:%M:%S')
        log_filename = get_result_filename(
                dataset,
                constraint_type,
                'train',
                'semisup',
                False,
                False)[:-4] + '.log'
        log_filename_full = os.path.join(cons_result_path,
                log_filename)
        file_handler=logging.FileHandler(log_filename_full, 'w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # log all the run parameters
        logger.info('----------Running New Instance----------')
        logger.info('dataset: %s, n_class: %d', dataset,n_class)
        logger.info('constraint type: %s, add MV const: %s',
                constraint_type, False)
        logger.info('labeled set: %s, use inequalities: %s',
                'train', False)

        run_wmrc_ds_comp(
                dataset_prefix,
                dataset_name=dataset,
                constraint_form=constraint_type,
                save_path=cons_result_path,
                logger=logger,
                )
