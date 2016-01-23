from os.path import expanduser

import numpy as np

from nilearn._utils.sparse_pca_utils import Experiment, \
    display_explained_variance_cluster, display_explained_variance_time, \
    display_density_time
from nilearn._utils.sparse_pca_utils import run
from nilearn.decomposition import SparsePCA


def adhd_20(n_jobs=1):
    estimators = [SparsePCA(alpha=alpha, batch_size=20,
                            random_state=0,
                            n_epochs=2,
                            warmup=False,
                            feature_ratio=feature_ratio)
                  for feature_ratio in np.linspace(1, 10, 4)
                  for alpha in np.logspace(-4, -1, 4)]
    estimators = estimators
    experiment = Experiment('hcp',
                            n_subjects=100,
                            smoothing_fwhm=4,
                            dict_init='rsn70',
                            output_dir=expanduser('~/output/sparse_pca'),
                            cachedir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=1,
                            n_jobs=n_jobs,
                            parallel_exp=True,
                            n_runs=1)
    output_dir = run(estimators, experiment)
    # gather_results(output_dir=expanduser('~/share/output/'
    #                                      'spca_cluster/2016-01-11_15-32-09/'
    #                                      'results/'))
    return output_dir


if __name__ == '__main__':
    # warnings.filterwarnings('ignore', category=DeprecationWarning)
    # adhd_20(n_jobs=32)
    # density('/storage/workspace/amensch/output/sparse_pca/2016-01-19_23-26-40')
    display_explained_variance_cluster('/home/arthur/drago/output/sparse_pca/2016-01-19_23-26-40')
    display_explained_variance_time('/home/arthur/drago/output/sparse_pca/2016-01-19_23-26-40')
    display_density_time('/home/arthur/drago/output/sparse_pca/2016-01-19_23-26-40')