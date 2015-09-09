from os.path import expanduser
from nilearn._utils.sparse_pca_utils import run, gather_results, display_all, \
    analyse, analyse_median_maps, plot_median
from nilearn.decomposition import SparsePCA
from nilearn._utils.sparse_pca_utils import Experiment


def adhd_20(n_jobs=1):
    ref_estimator = SparsePCA(alpha=0.05, batch_size=20,
                            reduction_method='none',
                            random_state=0,
                            support=False,
                            feature_ratio=1)
    estimators = [SparsePCA(alpha=0.05, batch_size=20,
                                 reduction_method='none',
                                 random_state=0,
                                 support=support,
                                 feature_ratio=feature_ratio)
                       for feature_ratio in [5, 10]
                       for support in [True, False]]
    estimators = [ref_estimator] + estimators
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
                            n_epochs=3,
                            n_runs=2, )
    output_dir = run(estimators, experiment)
    # output_dir = '/volatile/arthur/output/2015-11-27_13-56-35'
    # gather_results(output_dir=output_dir)
    # analyse(experiment, output_dir, n_jobs=32)
    # analyse_median_maps(output_dir)
    # plot_median(output_dir)
    # display_all(output_dir)
    return output_dir


if __name__ == '__main__':
    adhd_20(n_jobs=16)
