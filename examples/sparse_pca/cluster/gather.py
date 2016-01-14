import fnmatch
from os.path import join, expanduser
import os

import numpy as np
import pandas as pd
import json


def gather_results(job_dir, output_dir):
    full_dict_list = []
    for dirpath, dirname, filenames in os.walk(job_dir):
        for filename in fnmatch.filter(filenames, 'exp_*.json'):
            exp_num, ext = os.path.splitext(filename)
            with open(join(dirpath, filename), 'r') as f:
                exp_dict = json.load(f)
                exp_dict['score_test_file'] = join(output_dir,
                                                   'results', exp_num, 'debug/score_test.npy')
                exp_dict['score_test'] = np.load(
                        exp_dict['score_test_file'])[-1]
                exp_dict['density_file'] = join(output_dir,
                                                   'results', exp_num, 'debug/density.npy')
                exp_dict['density'] = np.load(exp_dict['density_file'])[-1]

                full_dict_list.append(exp_dict)

    results = pd.DataFrame(full_dict_list, columns=['feature_ratio',
                                                    'alpha',
                                                    'slice',
                                                    'random_state',
                                                    'score',
                                                    'density',
                                                    'score_test_file',
                                                    'density_file'])

    results.sort_values(by=['feature_ratio',
                            'alpha',
                            'slice',
                            'random_state'], inplace=True)
    results.to_csv(join(output_dir, 'results.csv'))

if __name__ == '__main__':
    gather_results(expanduser("~/share/output/spca_cluster/2016-01-11_15-32-09/jobs"),
                   expanduser("~/share/output/spca_cluster/2016-01-11_15-32-09/results"))