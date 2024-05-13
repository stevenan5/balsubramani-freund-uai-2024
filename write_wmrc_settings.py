import os
import json
from copy import deepcopy

if __name__ == '__main__':
    prefix= './datasets'

    # store relevant kwarg combinations in dictionaries
    semisup_kwargs = {
            'labeled_set' : 'valid',
            'bound_method' : 'binomial',
            'use_inequality_consts' : True,
            'add_mv_const': False,
            # 'get_confidences' : True,
            'get_confidences' : False,
            'n_max_labeled' : [100, 150, 200, 250, 300],
            }

    oracle_kwargs = {
            'labeled_set' : 'train',
            # doesn't matter what this is because we're not using any intervals
            'bound_method' : 'binomial',
            'use_inequality_consts' : False,
            # 'get_confidences' : True,
            'get_confidences' : False,
            'add_mv_const': False,
            # using empirical training accuracies is just for the transductive
            # setting and not meant for the inductive one
            'use_test' : False,
            'n_max_labeled' : [-1],
            }
    # list out which general dictionaries to use
    dics_to_use = [oracle_kwargs, semisup_kwargs]

    ### wrench datasets
    aa2_dic = {
            'dataset_name': 'aa2',
            'n_classes': 2,
            'use_test' : True,
            'pattern_neighborhood_size': 20,
            'pred_prob_incr' : 20,
            # there are specific situations where this is ignored, specifically
            # when oracle accuracies are used, stuff is getting replot, or all
            # validation data is used.
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    aa2_dics = [dict(gen_dic, **aa2_dic) for gen_dic in dics_to_use]
    aa2_dics[1]['n_max_labeled'] = [100, 113, 125, 137, 150, 161, 172]
    write_path = os.path.join(prefix, 'aa2_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(aa2_dics, fout)

    basketball_dic = {
            'dataset_name': 'basketball',
            'n_classes': 2,
            'use_test' : True,
            'pattern_neighborhood_size': 2,
            # doesn't matter how small this is, the rules basically get no
            # weight and the class frequency constraints are doing all the
            # heavy lifting
            'pred_prob_incr' : 5,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    basketball_dics = [dict(gen_dic, **basketball_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'basketball_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(basketball_dics, fout)

    breast_cancer_dic = {
            'dataset_name': 'breast_cancer',
            'n_classes': 2,
            'use_test' : True,
            'pattern_neighborhood_size': 2,
            'pred_prob_incr' : 10,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    breast_cancer_dics = [dict(gen_dic, **breast_cancer_dic) for gen_dic in dics_to_use]
    breast_cancer_dics[1]['n_max_labeled'] = [100, 121, 142, 163, 184, 206, 227]
    write_path = os.path.join(prefix, 'breast_cancer_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(breast_cancer_dics, fout)

    cardio_dic = {
            'dataset_name': 'cardio',
            'n_classes': 2,
            'use_test' : True,
            'pattern_neighborhood_size': 0,
            'pred_prob_incr' : 20,
            'n_runs' : 10,
            # don't compute by patterns since there's essentially no point
            'reject_threshold': 0,
            }
    # we want dataset specific entries to overwrite the general entries
    cardio_dics = [dict(gen_dic, **cardio_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'cardio_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(cardio_dics, fout)

    imdb_dic = {
            'dataset_name': 'imdb',
            'n_classes': 2,
            'use_test' : True,
            # first group is 50/50 even when the below is 2
            'pattern_neighborhood_size': 3,
            'pred_prob_incr' : 20,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    imdb_dics = [dict(gen_dic, **imdb_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'imdb_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(imdb_dics, fout)

    obs_dic = {
            'dataset_name': 'obs',
            'n_classes': 2,
            'use_test' : True,
            'pattern_neighborhood_size': 2,
            'pred_prob_incr' : 20,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    obs_dics = [dict(gen_dic, **obs_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'obs_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(obs_dics, fout)

    sms_dic = {
            'dataset_name': 'sms',
            'n_classes': 2,
            'use_test' : True,
            'pattern_neighborhood_size': 5,
            # also doesn't really matter since when intervals are estimated
            # the class frequencies are almost always the only non-zero weights
            'pred_prob_incr' : 10,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    sms_dics = [dict(gen_dic, **sms_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'sms_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(sms_dics, fout)

    yelp_dic = {
            'dataset_name': 'yelp',
            'n_classes': 2,
            'use_test' : True,
            'pattern_neighborhood_size': 6,
            'pred_prob_incr' : 10,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    yelp_dics = [dict(gen_dic, **yelp_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'yelp_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(yelp_dics, fout)

    youtube_dic = {
            'dataset_name': 'youtube',
            'n_classes': 2,
            'use_test' : True,
            'pattern_neighborhood_size': 4,
            'pred_prob_incr' : 20,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    youtube_dics = [dict(gen_dic, **youtube_dic) for gen_dic in dics_to_use]
    youtube_dics[1]['n_max_labeled'] = [100, 103, 107, 110, 113, 117, 120]
    write_path = os.path.join(prefix, 'youtube_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(youtube_dics, fout)

    domain_dic = {
            'dataset_name': 'domain',
            'n_classes': 5,
            'use_test' : True,
            'pattern_neighborhood_size': 4,
            'pred_prob_incr' : 50,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    domain_dics = [dict(gen_dic, **domain_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'domain_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(domain_dics, fout)
