from os.path import expanduser

from nilearn._utils.sparse_pca_utils import Experiment
from nilearn._utils.sparse_pca_utils import run
from nilearn.decomposition import SparsePCA


def adhd_20(n_jobs=1):
    ref_estimator = SparsePCA(alpha=0.01, batch_size=20,
                              reduction_method='none',
                              random_state=0,
                              # support=False,
                              feature_ratio=1)
    estimators = [SparsePCA(alpha=alpha, batch_size=20,
                            reduction_method='none',
                            random_state=0,
                            n_epochs=feature_ratio * 2,
                            feature_ratio=feature_ratio)
                  for feature_ratio in range(1, 2)
                  for alpha in [0.001, 0.005, 0.01, 0.05]]
    # for support in [True, False]]
    # estimators = [ref_estimator] + estimators
    estimators = estimators
    experiment = Experiment('adhd',
                            n_subjects=40,
                            smoothing_fwhm=6,
                            dict_init='rsn70',
                            output_dir=expanduser('~/output/sparse_pca'),
                            cachedir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=1,
                            n_jobs=4,
                            parallel_exp=True,
                            # n_epochs=5,
                            n_runs=1)
    output_dir = run(estimators, experiment)
    # output_dir = '/volatile/arthur/output/2015-11-27_13-56-35'
    # gather_results(output_dir=output_dir)
    # analyse(experiment, output_dir, n_jobs=32)
    # analyse_median_maps(output_dir)
    # plot_median(output_dir)
    # display_all(output_dir)
    return output_dir


if __name__ == '__main__':
    adhd_20(n_jobs=4)
