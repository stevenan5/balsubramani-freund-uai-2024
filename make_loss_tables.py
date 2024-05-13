import os
import copy
import numpy as np
import scipy.io as sio
import latextable
from texttable import Texttable

def add_bold(val):
    return "\\textbf{{{value:.2f}}}".format(value=val)

if __name__ == "__main__":

    print('Make sure you are out of the `wrench` environment and '
            'latextable=1.0.0 is installed.')

    labeled_datasets = True
    # labeled_datasets = False

    if labeled_datasets:
        prefix = 'labeled_'
    else:
        prefix = 'unlabeled_'

    if labeled_datasets:
        # Cancer is Breast Cancer
        datasets = ['AwA', 'Basketball', 'Cancer', 'Cardio', 'Domain',\
                'IMDB', 'OBS', 'SMS', 'Yelp', 'Youtube']
        methods = ['MV', 'OCDS', 'DP', 'EBCC', 'HyperLM', 'AMCL CC'\
                , 'BF' , '$d(\eta, g^{*})$']
    else:
        raise NotImplementedError

    # load all the data from the saved files
    results_folder_path = './results'

    bs_ll_table = Texttable()
    err_table = Texttable()
    bs_ll_table_rows = []
    err_table_rows = []

    for rows in [bs_ll_table_rows, err_table_rows]:
        for method in methods:
            rows.append([method])

    if labeled_datasets:
        fn = 'loss_labeled_ttest.mat'
    else:
        fn = 'loss_unlabeled_ttest.mat'

    mdic = sio.loadmat(os.path.join(results_folder_path, fn))
    bs = mdic['brier_score_train']
    ll = mdic['log_loss_train']
    zo = mdic['err_train']
    bs_bold = mdic['brier_score_train_ttest']
    ll_bold = mdic['log_loss_train_ttest']
    zo_bold = mdic['err_train_ttest']


    ### make log loss table
    # add row for WMRC oracle
    ll = np.vstack((ll, mdic['wmrc_oracle_log_loss_train'].squeeze()))
    ll_bold = np.vstack((ll_bold, np.zeros(bs_bold.shape[1])))

    for d_ind, dataset in enumerate(datasets):
        # make log_loss loss table
        for i, row in enumerate(err_table_rows):

            val_ll = np.round(ll[i, d_ind], 2)
            # need $ because for some reason latextable will always print 3
            # digits after the . even though the string is xx.xx
            app_value = add_bold(val_ll) if ll_bold[i, d_ind]\
                    else "${value:.2f}$".format(value=val_ll)
            row.append(app_value)

    # for row in err_table_rows:
    #     print(row)

    err_header = copy.deepcopy(datasets)
    err_header.insert(0, 'Method')
    err_table_rows.insert(0, err_header)
    err_table.set_cols_align(["c"]* (1 + len(datasets)))
    err_table.add_rows(err_table_rows)
    err_caption = 'Log Loss Comparison'
    err_label = 'tab:' + prefix + 'wrench_log_loss'
    err_table_latex = latextable.draw_latex(err_table, use_booktabs=True, caption=err_caption, label=err_label)
    with open('results/' + prefix + 'log_loss_table.txt', 'w') as f:
        f.write(err_table_latex)
        f.close()


    ### make 0-1 loss/brier score table
    # add rows for WMRC oracle
    bs = np.vstack((bs, mdic['wmrc_oracle_brier_score_train'].squeeze()))
    bs_bold = np.vstack((bs_bold, np.zeros(bs_bold.shape[1])))
    # add row for WMRC oracle
    zo = np.vstack((zo, mdic['wmrc_oracle_err_train'].squeeze()))
    zo_bold = np.vstack((zo_bold, np.zeros(zo_bold.shape[1])))

    for d_ind, dataset in enumerate(datasets):
        # make brier score /log loss table
        for i, row in enumerate(bs_ll_table_rows):
            val_zo = np.round(zo[i, d_ind] * 100, 2)
            val_bs = np.round(bs[i, d_ind], 2)

            # need $ because for some reason latextable will always print 3
            # digits after the . even though the string is xx.xx
            app_value = add_bold(val_zo) if zo_bold[i, d_ind]\
                    else "${value:.2f}$".format(value=val_zo)
            row.append(app_value)
            app_value = add_bold(val_bs) if bs_bold[i, d_ind]\
                    else "${value:.2f}$".format(value=val_bs)
            row.append(app_value)

    # make sub header which is just '0-1' and 'BS' alternating
    subheader = [""] + ['0-1', 'BS'] * len(datasets)
    bs_ll_table_rows.insert(0, subheader)
    # make header
    header = [("Method", 1)]
    for i, dataset in enumerate(datasets):
        header.append((dataset, 2))

    bs_ll_table.set_cols_align(["c"] * (1 + 2 * len(datasets)))
    bs_ll_table.add_rows(bs_ll_table_rows)
    bs_ll_caption = '0-1 Loss, Brier Score Comparison'
    bs_ll_label = 'tab:' + prefix + 'wrench_zero_one_brier_score'
    bs_ll_table_latex = latextable.draw_latex(bs_ll_table, use_booktabs=True, caption=bs_ll_caption, label=bs_ll_label, multicolumn_header=header)
    with open('results/' + prefix + 'zero_one_brier_score_table.txt', 'w') as f:
        f.write(bs_ll_table_latex)
        f.close()

    if labeled_datasets:
        ### make a table for avg # datapoints used, avg # of classifiers available
        wmrc_stats_table = Texttable()

        # get data
        wmrc_avg_labeled = mdic['wmrc_avg_labeled_pts_per_classifier'].squeeze().tolist()
        wmrc_avg_avail_lf = mdic['wmrc_avg_num_classifier_available'].squeeze().tolist()
        all_avail_lf = mdic['num_classifier_per_dataset'].squeeze().tolist()

        if not isinstance(wmrc_avg_labeled, list):
            wmrc_avg_labeled = [wmrc_avg_labeled]
        if not isinstance(wmrc_avg_avail_lf, list):
            wmrc_avg_avail_lf = [wmrc_avg_avail_lf]
        if not isinstance(all_avail_lf, list):
            all_avail_lf = [all_avail_lf]

        # make row of labeled points used (with percent)
        lab_pts_row = ['Avg.~Labeled Pts per LF']
        # nl for num labeled
        for i, nl in enumerate(wmrc_avg_labeled):
            app_value = "$100$" if nl == 100\
                    else "${value:.2f}$".format(value=nl )
            lab_pts_row.append(app_value)

        # make row for number of available classifiers (with percent)
        n_lf_row = ['Avg.~LFs Available']
        for i, nlf in enumerate(wmrc_avg_avail_lf):
            n_all_lf = all_avail_lf[i]
            app_value = str(nlf) if nlf == n_all_lf\
                    else "${value:.2f} ({pct:.2f}\\%)$".format(value=nlf, pct = 100 * nlf/n_all_lf)
            n_lf_row.append(app_value)

        # make header
        wmrc_stats_header = copy.deepcopy(datasets)
        wmrc_stats_header.insert(0, '')

        wmrc_stats_table_rows = [wmrc_stats_header, lab_pts_row, n_lf_row]
        wmrc_stats_table.set_cols_align(["c"] * (1 + len(datasets)))
        wmrc_stats_table.add_rows(wmrc_stats_table_rows)
        wmrc_stats_caption = 'WMRC Statistics'
        wmrc_stats_label = 'tab:wmrc_stats'
        wmrc_stats_table_latex = latextable.draw_latex(wmrc_stats_table, use_booktabs=True, caption=wmrc_stats_caption, label=wmrc_stats_label)
        with open('results/' + 'wmrc_stats.txt', 'w') as f:
            f.write(wmrc_stats_table_latex)
            f.close()
