from os.path import join, expanduser
import numpy as np

from nilearn.decomposition import DictLearning
from nilearn.decomposition.isbi_utils import run, \
    analyse, analyse_incr, plot_incr, plot_full, \
    drop_memmmap, Experiment, convert_nii_to_pdf, \
    clean_memory, gather_results, analyse_median_maps


# Long experiment for stability
# estimators = []
# for compression_type in ['range_finder', 'subsample']:
#     for reduction_ratio in np.linspace(0.1, 1, 3):
#         for alpha in np.linspace(18, 26, 5):
#             estimators.append(DictLearning(alpha=alpha, batch_size=20,
#                                            compression_type=compression_type,
#                                            random_state=0,
#                                            forget_rate=1,
#                                            reduction_ratio=reduction_ratio))
# estimators.append(DictLearning(alpha=20, batch_size=20,
#                                compression_type='subsample',
#                                random_state=0,
#                                forget_rate=1,
#                                reduction_ratio=1))
# estimators.append(DictLearning(alpha=20, batch_size=20,
#                                compression_type='subsample',
#                                random_state=0,
#                                forget_rate=1,
#                                reduction_ratio=1))
# experiment = Experiment('adhd',
#                         n_subjects=40,
#                         smoothing_fwhm=6,
#                         dict_init='rsn20',
#                         output_dir=expanduser('~/output'),
#                         cache_dir=expanduser('~/nilearn_cache'),
#                         data_dir=expanduser('~/data'),
#                         n_slices=1,
#                         n_jobs=5,
#                         n_epochs=1,
#                         # Out of core dictionary learning specifics
#                         temp_folder=expanduser('~/temp'),
#                         # Stability specific
#                         n_runs=100)

def adhd_20():
# ADHD RSN20 intensive experiment
    estimators = []

    estimators.append(DictLearning(alpha=4, batch_size=20,
                                   compression_type='subsample',
                                   random_state=0,
                                   forget_rate=1,
                                   reduction_ratio=1))
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
                            n_subjects=40,
                            smoothing_fwhm=6,
                            dict_init=expanduser('~/ica/canica_resting_state'
                                                 '_20.nii.gz'),
                            output_dir=expanduser('~/output'),
                            cache_dir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=1,
                            n_jobs=12,
                            n_epochs=1,
                            temp_folder=expanduser('~/temp'),
                            n_runs=1)

    # temp_folder = drop_memmmap(estimators, experiment)
    temp_folder = '/volatile/arthur/temp/2015-10-12_16-29-09'
    # output_dir = run(estimators, experiment, temp_folder=temp_folder)
    output_dir = '/volatile/arthur/drago_output/2015-10-14_21-02-57'
    # output_dir = expanduser('/volatile/arthur/work/output/test/2015-10-06_13-04-14')
    # gather_results(output_dir=output_dir)
    # analyse(output_dir, n_jobs=12, limit=10)
    # analyse_incr(output_dir, n_jobs=12, n_run_var=1)
    plot_full(output_dir)
    plot_incr(output_dir, 0.1)
    analyse_median_maps(output_dir)
    # convert_nii_to_pdf(join(output_dir, 'stability'), n_jobs=15)


def hcp_70():
    # HCP RSN70 explorative experiment
    estimators = []
    estimators.append(DictLearning(alpha=4, batch_size=20,
                                   compression_type='subsample',
                                   random_state=0,
                                   forget_rate=1,
                                   reduction_ratio=1))
    for compression_type in ['range_finder', 'subsample']:
        for reduction_ratio in np.linspace(0.1, 1, 10):
            for alpha in np.linspace(1, 7, 7):
                estimators.append(DictLearning(alpha=alpha, batch_size=20,
                                               compression_type=compression_type,
                                               random_state=0,
                                               forget_rate=1,
                                               reduction_ratio=reduction_ratio))
    # estimators = []
    # for alpha in np.linspace(1, 10, 10):
    #     estimators.append(DictLearning(alpha=alpha, batch_size=20,
    #                                    compression_type='subsample',
    #                                    random_state=0,
    #                                    forget_rate=1,
    #                                    reduction_ratio=1))
    experiment = Experiment('hcp_reduced',
                            n_subjects=70,
                            smoothing_fwhm=6,
                            dict_init='rsn70',
                            output_dir=expanduser('~/output'),
                            cache_dir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=1,
                            n_jobs=16,
                            n_epochs=1,
                            # Out of core dictionary learning specifics
                            temp_folder=expanduser('~/temp'),
                            # Stability specific
                            n_runs=10)
    # temp_folder = '/home/parietal/amensch/temp/2015-10-12_17-06-34'
    # output_dir = run(estimators, experiment, temp_folder=temp_folder)
    output_dir = expanduser('~/output/2015-10-14_23-46-52')
    gather_results(output_dir)
    analyse(output_dir, n_jobs=4, limit=1)
    analyse_incr(output_dir, n_jobs=4,
                 n_run_var=1)

adhd_20()