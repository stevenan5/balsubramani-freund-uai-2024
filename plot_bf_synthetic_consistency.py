import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import savemat
from wrench.labelmodel import WMRC

if __name__ == "__main__":
    fn_end = '_accuracy_semisup_trainlabels_eqconst.mat'

    # load all the data from the saved files
    results_folder_path = './results/synthetic_dawid_skene'

    enlarge = False
    # enlarge = True

    if enlarge:
        plt.rcParams.update({'font.size': 20})

    n_pts = [100, 1000, 10000, 100000]
    # change dataset path if using synthetic datasets
    # create dataset names for synthetic datasets
    fn_part = lambda x: 'synth_3p_' + str(x) + 'n_100nval__'
    n_synth = 10

    res = np.zeros((n_synth, len(n_pts)))

    for i in range(n_synth):
        for j, n_pt in enumerate(n_pts):
            res_name = 'WMRC_' + fn_part(n_pt) + str(i) + fn_end
            res_path = os.path.join(results_folder_path, 
                    fn_part(n_pt) + str(i) + '/WMRC/accuracy/')
            res_path = os.path.join(res_path, res_name)
            mdic = sio.loadmat(res_path)
            res[i, j] = mdic['kl_from_ds']


    # get the mean and max/min values
    kl_mean = res.mean(axis=0)
    kl_min = res.min(axis=0)
    kl_max = res.max(axis=0)

    fig, ax = plt.subplots()
    # fig.dpi = 1200
    suffix = '_enlarged' if enlarge else ''
    plot_fn = os.path.join(results_folder_path, 'WMRC_kl_from_ds' + suffix + '.pdf')
    ax.plot(n_pts, kl_mean, '-', color='blue', label='Mean KL Div.')
    ax.fill_between(n_pts, kl_min, kl_max, color='cornflowerblue', alpha=0.3,
        label='Max/Min KL Div.')
    plt.xscale('log')
    plt.yscale('log')
    # ax.plot(n_labeled_cts, logloss_mean, 'ro-', label='Log Loss')
    # ax.fill_between(n_labeled_cts, logloss_2_5, logloss_97_5, color='r', alpha=0.3)
            # label='95% Confidence')
    # ax.plot(n_labeled_cts, brier_mean, 'bo-', label='Brier Score')
    # ax.fill_between(n_labeled_cts, brier_2_5, brier_97_5, color='b', alpha=0.3)
            # label='95% Confidence')

    # x_min = min(n_labeled_cts)
    # x_max = max(n_labeled_cts)
    # ax.hlines(1 - oracle_acc, x_min, x_max, 'g', label='Oracle 0-1')
    # ax.hlines(oracle_logloss, x_min, x_max, 'r', label='Oracle Log Loss')
    # ax.hlines(oracle_brier, x_min, x_max, 'b', label='Oracle Brier Score')

    ax.legend()
    ax.set_ylabel('KL Divergence')
    ax.set_xlabel('# Points')

    fig.savefig(plot_fn, bbox_inches='tight', format='pdf')
    plt.close(fig)
