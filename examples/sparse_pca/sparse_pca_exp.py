import warnings
from os.path import expanduser

from nilearn._utils.sparse_pca_utils import Experiment, gather_results, \
    display_explained_variance, convert_nii_to_pdf
from nilearn._utils.sparse_pca_utils import run
from nilearn.decomposition import SparsePCA

import numpy as np


def adhd_20(n_jobs=1):
    estimators = [SparsePCA(alpha=alpha, batch_size=20,
                            random_state=0,
                            n_epochs=2,
                            warmup=False,
                            feature_ratio=feature_ratio)
                  for feature_ratio in np.linspace(1, 10, 3)
                  for alpha in np.logspace(-4, -1, 4)]
    estimators = estimators
    experiment = Experiment('adhd',
                            n_subjects=40,
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
    gather_results(output_dir=output_dir)
    return output_dir


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    adhd_20(n_jobs=10)
