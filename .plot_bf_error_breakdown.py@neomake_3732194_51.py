import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import savemat
from wrench.labelmodel import WMRC
from matplotlib.patches import Patch
import matplotlib.transforms

if __name__ == "__main__":
    datasets = ['aa2', 'basketball', 'breast_cancer', 'cardio', 'domain',\
            'imdb', 'obs', 'sms', 'yelp', 'youtube']
    datasets = ['sms']

    enlarge = False
    # enlarge = True
    include_ds = True
    # include_ds = False

    if enlarge:
        plt.rcParams.update({'font.size': 20})
    wmrc_names = ['WMRC']
    wmrc_fn_ends = ['_accuracy_semisup_validlabels_ineqconst_binomial.mat']

    # load all the data from the saved files
    results_folder_path = './results'

    for dataset in datasets:

        dataset_result_path = os.path.join(results_folder_path, dataset)

        if dataset == 'aa2':
            n_labeled_cts = [100, 113, 125, 137, 150, 161, 172]
        elif dataset == 'breast_cancer':
            n_labeled_cts = [100, 121, 142, 163, 184, 206, 227]
        elif dataset == 'youtube':
            n_labeled_cts = [100, 103, 107, 110, 113, 117, 120]
        else:
            n_labeled_cts = [100, 150, 200, 250, 300]

        num_pts = len(n_labeled_cts)

        logloss_mean = np.zeros(num_pts)
        logloss_min = np.zeros(num_pts)
        logloss_max = np.zeros(num_pts)
        eps_l_inf = np.zeros(num_pts)

        # load oracle losses
        oracle_result_fn = 'WMRC_' + dataset + '_accuracy_semisup_trainlabels'+\
                '_eqconst.mat'
        oracle_wmrc_full = os.path.join(dataset_result_path, 'WMRC/accuracy/' +\
                    oracle_result_fn)

        oracle_mdic = sio.loadmat(oracle_wmrc_full)
        oracle_logloss = oracle_mdic['log_loss_train']
        oracle_logloss_vec = np.ones(num_pts) * oracle_logloss

        # load Dawid-Skene log loss
        ds_full = os.path.join(dataset_result_path, 'DawidSkene/one_coin/' +\
                    'DawidSkene_' + dataset + '.mat')
        ds_mdic = sio.loadmat(ds_full)
        ds_log_loss = np.ones(num_pts) * ds_mdic['log_loss_train']
        ds_log_loss_oracle = np.ones(num_pts) * ds_mdic['log_loss_train_oracle']

        fig, ax = plt.subplots()
        # fig.dpi = 1200
        suffix = '_enlarged' if enlarge else ''
        if include_ds:
            plot_fn = os.path.join(dataset_result_path,\
                    'WMRC_' + dataset + '_approx_uncert_acc_one_coin' + suffix + '.pdf')
        else:
            plot_fn = os.path.join(dataset_result_path,\
                    'WMRC_' + dataset + '_approx_uncert_acc_no_ds' + suffix + '.pdf')

        wmrc_name = 'WMRC/accuracy/'
        wmrc_fn_end = '.mat'

        for i, n_labeled in enumerate(n_labeled_cts):
            result_fn = 'WMRC_' + dataset + '_accuracy_semisup_' +\
                    str(n_labeled) + 'validlabels_ineqconst_binomial' + wmrc_fn_end
            wmrc_fn_full = os.path.join(dataset_result_path, wmrc_name +\
                    result_fn)

            # load data
            mdic = sio.loadmat(wmrc_fn_full)

            logloss_min[i] = np.min(mdic['log_loss_train'])
            logloss_mean[i] = mdic.get('log_loss_train_mean', mdic['log_loss_train'])
            logloss_max[i] = np.max(mdic['log_loss_train'])
            eps_l_inf[i] = np.mean(mdic['epsilon_l_inf'])

        # do some post processing because we're going to stack the lines
        for i in range(len(n_labeled_cts)):
            logloss_max[i] -= logloss_mean[i]
            logloss_mean[i] -= logloss_min[i]
            logloss_min[i] -= oracle_logloss_vec[0][0]

        # handles, _ = ax.get_legend_handles_labels()
        # labels = [r'Model Unc. $d(\eta, g^{*})$', r'Min BF Appr. Unc. $d(g^{*}, g^{bf})$', r'Mean BF Appr. Unc. $d(g^{*}, g^{bf})$', r'Max BF Appr. Unc. $d(g^{*}, g^{bf})$']
        labels = [r'$d(\eta, g^{*})$', r'Min $d(g^{*}, g^{bf})$', r'Mean $d(g^{*}, g^{bf})$', r'Max $d(g^{*}, g^{bf})$']
        if enlarge:
            labels[0] = ''
        # colors = ['mistyrose', 'tomato', 'red', 'firebrick']
        colors = [ 'green', 'cornflowerblue', 'tomato', 'gold']
        # colors = [ 'cornflowerblue', 'gold', 'green', 'tomato' ]
        handles = [Patch(facecolor=color, edgecolor='black', label=label) for color, label in zip(colors, labels)]
        ax.stackplot(n_labeled_cts, oracle_logloss_vec, logloss_min, logloss_mean, logloss_max,
                labels=labels, colors=colors)

        lw = 3 if enlarge else 1
        ax.axhline(oracle_logloss_vec.squeeze()[0], c='k', lw=lw, ls='-')
        ax.axhline(ds_log_loss_oracle.squeeze()[0], c='k', lw=lw, ls='--')

        if include_ds:
            ax.axhline(ds_log_loss.squeeze()[0], c='k', lw=lw, ls=':')
            # ax.plot(n_labeled_cts, ds_log_loss.squeeze(), 'b-')
            # ax.plot(n_labeled_cts, oracle_logloss_vec.squeeze(), '-', c='chartreuse')
            # ax.plot(n_labeled_cts, oracle_logloss_vec.squeeze(), 'g-')
            # ax.plot(n_labeled_cts, ds_log_loss_oracle.squeeze(), 'k-')

        # force the x limit to be at the bounds of labeled data
        ax.set_xlim([n_labeled_cts[0], n_labeled_cts[-1]])

        # Create a second y-axis
        y2 = ax.twinx()
        # Copy the y limits from the left axis
        ylims = ax.get_ylim()
        y2.set_ylim(ylims)
        mirror_ytick_positions = [oracle_logloss[0,0], ds_log_loss_oracle[0,0]]
        y2_tick_labels = [r'$d(\eta, g^{*})$', r'$d(\eta, g^{ds*})$']
        if include_ds:
            mirror_ytick_positions.append(ds_log_loss[0,0])
            y2_tick_labels.append( r'$d(\eta, g^{ds})$')
        # Set the right y-yick positions
        y2.set_yticks(mirror_ytick_positions)
        # Set the right y-tick labels
        y2.set_yticklabels(y2_tick_labels)

        # if the model uncertainty and DS oracle log loss are too close,
        # move them apart
        pct_dist = (mirror_ytick_positions[1] - mirror_ytick_positions[0]) / (ylims[1] - ylims[0])
        scale = 2.5 if enlarge else 1
        if pct_dist < 0.08:
            dy_ds = 2/72. * scale
            dy_mu = -2/72. * scale
            if pct_dist < 0.01:
                dy_ds = 5/72. * scale
                dy_mu = -5/72. * scale
            ds_orac_offset = matplotlib.transforms.ScaledTranslation(0, dy_ds, fig.dpi_scale_trans)
            model_un_offset = matplotlib.transforms.ScaledTranslation(0, dy_mu, fig.dpi_scale_trans)

            # apply offset transform to all x ticklabels.
            for i, label in enumerate(y2.yaxis.get_majorticklabels()):
                if i == 0:
                    label.set_transform(label.get_transform() + model_un_offset)
                elif i == 1:
                    label.set_transform(label.get_transform() + ds_orac_offset)

        _, labels = ax.get_legend_handles_labels()
        # move legend position based on how big model uncertainty is
        if mirror_ytick_positions[0]/ (ylims[1] - ylims[0]) > 0.35:
            legend_loc = 'lower right'
        else:
            legend_loc = 'upper right'

        # manually change the legend for figures going into the main paper
        if dataset == 'aa2':
            if legend_loc == 'upper right':
                ax.legend(handles[::-1], labels[::-1], loc=legend_loc, bbox_to_anchor=(0.98,0.95), framealpha=0.89)
                if enlarge:
                    prop = {'size': 15}
                    ax.legend(handles[::-1], labels[::-1], loc=legend_loc, bbox_to_anchor=(0.100,0.96), framealpha=0.89, prop=prop)

        elif dataset == 'sms' and legend_loc=='upper right':
            if enlarge:
                ax.legend(handles[::-1], labels[::-1], loc=legend_loc, bbox_to_anchor=(0.98,0.97), framealpha=0.89)
            else:
                ax.legend(handles[::-1], labels[::-1], loc=legend_loc, bbox_to_anchor=(1,0.95), framealpha=0.89)
        else:
            ax.legend(handles[::-1], labels[::-1], loc=legend_loc, framealpha=0.89)
        ax.set_ylabel('Log Loss')
        # ax.set_xlabel(r'Avg. $\|\|\epsilon\|\|_\infty$, # Labeled Points')
        ax.set_xlabel('# Labeled Points')

        fig.savefig(plot_fn, bbox_inches='tight', format='pdf')
        plt.close(fig)
