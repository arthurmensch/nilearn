import warnings
from os.path import expanduser

from nilearn._utils.sparse_pca_utils import Experiment, gather_results, \
    display_explained_variance
from nilearn._utils.sparse_pca_utils import run
from nilearn.decomposition import SparsePCA

import numpy as np


def adhd_20(n_jobs=1):
    ref_estimator = SparsePCA(alpha=0.01, batch_size=20,
                              random_state=0,
                              feature_ratio=1)
    estimators = [SparsePCA(alpha=alpha, batch_size=20,
                            random_state=0,
                            n_epochs=1,
                            feature_ratio=feature_ratio)
                      for feature_ratio in np.linspace(1, 10, 3)
                      for alpha in np.logspace(-5, -1, 5)]
    # for support in [True, False]]
    # estimators = [ref_estimator] + estimators
    estimators = estimators
    experiment = Experiment('hcp',
                            n_subjects=10,
                            smoothing_fwhm=4,
                            dict_init='rsn70',
                            output_dir=expanduser('~/output/sparse_pca'),
                            cachedir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=1,
                            n_jobs=n_jobs,
                            parallel_exp=True,
                            # n_epochs=5,
                            n_runs=1)
    output_dir = run(estimators, experiment)
    output_dir = '/home/arthur/drago/output/sparse_pca/2015-12-29_14-05-13'
    gather_results(output_dir=output_dir)
    # analyse(experiment, output_dir, n_jobs=32)
    # analyse_median_maps(output_dir)
    # plot_median(output_dir)
    # display_all(output_dir)
    return output_dir


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    adhd_20(n_jobs=15)
    output_dir = '/home/arthur/drago/output/sparse_pca/2015-12-29_14-05-13'
    gather_results(output_dir)
    display_explained_variance(output_dir)