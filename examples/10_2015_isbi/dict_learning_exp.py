import os
from os.path import expanduser
import shutil

import numpy as np

from nilearn.decomposition import DictLearning
from isbi_utils import run, \
    analyse, analyse_num_exp, plot_num_exp, plot_full, \
    drop_memmmap, Experiment, gather_results, analyse_median_maps, \
    plot_median, plot_full_multiple


def adhd_20(n_jobs=1):
    # ADHD RSN20 intensive experiment
    estimators = [DictLearning(alpha=1, batch_size=20,
                               compression_type='subsample',
                               random_state=0,
                               forget_rate=1,
                               reduction_ratio=1)]

    for compression_type in ['range_finder', 'subsample']:
        for reduction_ratio in np.linspace(0.1, 1, 10):
            for alpha in np.linspace(2, 8, 4):
                estimators.append(DictLearning(alpha=alpha, batch_size=20,
                                               compression_type=
                                               compression_type,
                                               random_state=0,
                                               forget_rate=1,
                                               reduction_ratio=
                                               reduction_ratio))
    experiment = Experiment('adhd',
                            n_subjects=4,
                            smoothing_fwhm=6,
                            dict_init=expanduser('rsn20'),
                            output_dir=expanduser('~/output'),
                            cachedir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=2,
                            n_jobs=n_jobs,
                            n_epochs=1,
                            reference=False,
                            temp_folder=expanduser('~/temp'),
                            n_runs=48)
    temp_folder = drop_memmmap(estimators, experiment)
    output_dir = run(estimators, experiment, temp_folder=temp_folder)
    gather_results(output_dir=output_dir)
    analyse(output_dir, n_jobs=n_jobs)
    analyse_num_exp(output_dir, n_jobs=n_jobs, n_run_var=4)
    analyse_median_maps(output_dir)
    plot_num_exp(output_dir)
    plot_full(output_dir, n_exp=9)
    plot_median(output_dir)
    return output_dir


def hcp_70(n_jobs=1):
    # HCP RSN70 explorative experiment
    estimators = [DictLearning(alpha=6, batch_size=20,
                               compression_type='subsample',
                               random_state=0,
                               forget_rate=1,
                               reduction_ratio=1)]

    for compression_type in ['range_finder', 'subsample']:
        for compression_ratio in np.concatenate([np.array([0.01, 0.025,
                                                           0.5, 0.075]),
                                                 np.linspace(0.1, 1, 10)]):
            for alpha in np.linspace(1, 10, 10):
                estimators.append(DictLearning(alpha=alpha, batch_size=20,
                                               compression_type=
                                               compression_type,
                                               random_state=0,
                                               forget_rate=1,
                                               reduction_ratio=
                                               compression_ratio))
    experiment = Experiment('hcp_reduced',
                            n_subjects=75,
                            smoothing_fwhm=6,
                            dict_init='rsn70',
                            output_dir=expanduser('~/output'),
                            cachedir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=1,
                            n_jobs=n_jobs,
                            n_epochs=1,
                            reference=False,
                            # Out of core dictionary learning specifics
                            temp_folder=expanduser('~/temp'),
                            # Stability specific
                            n_runs=9)
    temp_folder = drop_memmmap(estimators, experiment)
    output_dir = run(estimators, experiment, temp_folder=temp_folder)
    gather_results(output_dir)
    analyse(experiment, output_dir, n_jobs=n_jobs, limit=9)
    analyse_num_exp(experiment, output_dir, n_jobs=n_jobs, limit=9,
                    n_run_var=3)
    analyse_median_maps(output_dir, reduction_ratio=0.025)
    plot_median(output_dir)
    plot_full(output_dir, n_exp=2)
    return output_dir


def main():
    n_jobs = 1
    try:
        os.makedirs(expanduser('~/temp'))
    except:
        pass
    try:
        os.makedirs(expanduser('~/output'))
    except:
        pass
    output_adhd = adhd_20(n_jobs=n_jobs)
    output_hcp = hcp_70(n_jobs=n_jobs)
    plot_full_multiple([output_adhd, output_hcp], n_exp_list=[9, 2])
    shutil.copytree(output_adhd, os.path.join(expanduser('~/figures'), 'adhd'))
    shutil.copytree(output_adhd, os.path.join(expanduser('~/figures'), 'hcp'))
