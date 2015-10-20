from os.path import join, expanduser
import numpy as np

from nilearn.decomposition import DictLearning
from nilearn.decomposition.isbi_utils import run, \
    analyse, analyse_num_exp, plot_num_exp, plot_full, \
    drop_memmmap, Experiment, convert_nii_to_pdf, \
    clean_memory, gather_results, analyse_median_maps, plot_median


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
#                         cachedir=expanduser('~/nilearn_cache'),
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

    estimators.append(DictLearning(alpha=1, batch_size=20,
                                   compression_type='subsample',
                                   random_state=0,
                                   forget_rate=1,
                                   reduction_ratio=1))
    # for compression_type in ['range_finder', 'subsample']:
    #     for reduction_ratio in np.linspace(0.1, 1, 10):
    #         for alpha in np.linspace(2, 8, 4):
    #             estimators.append(DictLearning(alpha=alpha, batch_size=20,
    #                                            compression_type=
    #                                            compression_type,
    #                                            random_state=0,
    #                                            forget_rate=1,
    #                                            reduction_ratio=
    #                                            reduction_ratio))
    experiment = Experiment('adhd',
                            n_subjects=4,
                            smoothing_fwhm=6,
                            dict_init=expanduser('rsn20'),
                            output_dir=expanduser('~/output'),
                            cachedir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=2,
                            n_jobs=12,
                            n_epochs=1,
                            temp_folder=expanduser('~/temp'),
                            n_runs=1)

    # temp_folder = drop_memmmap(estimators, experiment)
    # temp_folder = '/volatile/arthur/temp/2015-10-15_17-38-44'
    output_dir = run(estimators, experiment)
    # output_dir = expanduser('~/output/2015-10-14_21-02-57')
    # analyse_median_maps(output_dir)
    gather_results(output_dir=output_dir)
    # analyse(output_dir, n_jobs=28, limit=12)
    # analyse_num_exp(output_dir, n_jobs=20, n_run_var=4)
    # analyse_median_maps(output_dir)
    # plot_num_exp(output_dir)
    # plot_full(output_dir)
    # plot_num_exp(output_dir, reduction_ratio_list=[0.05, 0.2])
    # plot_full(output_dir)
    # plot_median(output_dir)
    # plot_num_exp(output_dir, 0.1)
    # convert_nii_to_pdf(join(output_dir, 'stability'), n_jobs=15)


def hcp_70():
    # HCP RSN70 explorative experiment
    estimators = []
    # alpha_list = np.array([[5, 6, 5, 5, 4, 4, 4, 4, 4, 4],
    #                        [2, 3, 3, 3, 4, 3, 3, 4, 4, 4]])

    # estimators.append(DictLearning(alpha=4, batch_size=20,
    #                                compression_type='subsample',
    #                                random_state=0,
    #                                forget_rate=1,
    #                                reduction_ratio=1))
    for compression_type in ['range_finder', 'subsample']:
        for reduction_ratio in [0.01, 0.025, 0.05, 0.075]:
            for alpha in [0.25, 0.5, 1, 2, 3]:
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
                            n_subjects=75,
                            smoothing_fwhm=6,
                            dict_init='rsn70',
                            output_dir=expanduser('~/output'),
                            cachedir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=1,
                            n_jobs=16,
                            n_epochs=1,
                            # Out of core dictionary learning specifics
                            temp_folder=expanduser('~/temp'),
                            # Stability specific
                            n_runs=20)
    # temp_folder = '/home/parietal/amensch/temp/2015-10-12_17-06-34'
    # temp_folder = drop_memmmap(estimators, experiment)
    # output_dir = run(estimators, experiment, temp_folder=temp_folder)
    output_dir = expanduser('~/output/2015-10-14_23-46-52')
    gather_results(output_dir)
    analyse(experiment, output_dir, n_jobs=10, limit=8)
    analyse_num_exp(experiment, output_dir, n_jobs=10,
                    n_run_var=2, limit=8)
    # plot_full(output_dir, n_exp=5)
    # plot_num_exp(output_dir, reduction_ratio_list=[0.1, 0.2], n_exp=5)
    # plot_full(output_dir)


def hcp_full_70():
    # HCP RSN70 explorative experiment
    estimators = []
    for alpha in np.linspace(1, 10, 10):
        estimators.append(DictLearning(alpha=alpha, batch_size=20,
                                       compression_type='subsample',
                                       random_state=0,
                                       forget_rate=1,
                                       reduction_ratio=1))
    experiment = Experiment('hcp',
                            n_subjects=40,
                            smoothing_fwhm=6,
                            dict_init='rsn70',
                            output_dir=expanduser('~/output'),
                            cachedir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=1,
                            n_jobs=10,
                            n_epochs=1,
                            # Out of core dictionary learning specifics
                            temp_folder=expanduser('~/temp'),
                            # Stability specific
                            n_runs=1)
    # temp_folder = expanduser('~/temp/2015-10-15_23-45-45')
    # temp_folder = drop_memmmap(estimators, experiment)
    # output_dir = run(estimators, experiment, temp_folder=temp_folder)
    output_dir = expanduser('~/output/2015-10-14_23-46-52')
    gather_results(output_dir)
    # analyse(experiment, output_dir, n_jobs=20, limit=1)
    # analyse_num_exp(output_dir, n_jobs=20,
    #                 n_run_var=1)

def hcp_rf_70():
    # HCP RSN70 explorative experiment
    estimators = []
    for reduction_ratio in [0.1, 0.2, 0.3, 0.4]:
        estimators.append(DictLearning(alpha=4, batch_size=20,
                                       compression_type='range_finder',
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
                            n_subjects=40,
                            smoothing_fwhm=6,
                            dict_init='rsn70',
                            output_dir=expanduser('~/output'),
                            cachedir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=1,
                            n_jobs=10,
                            n_epochs=1,
                            # Out of core dictionary learning specifics
                            temp_folder=expanduser('~/temp'),
                            # Stability specific
                            n_runs=10)
    temp_folder = '/home/parietal/amensch/temp/2015-10-12_17-06-34'
    # temp_folder = drop_memmmap(estimators, experiment)
    output_dir = run(estimators, experiment, temp_folder=temp_folder)
    # output_dir = expanduser('~/output/2015-10-14_23-46-52')
    gather_results(output_dir)
    analyse(experiment, output_dir, n_jobs=20, limit=3)
    analyse_num_exp(output_dir, n_jobs=20,
                    n_run_var=1, limit=3)

# adhd_20()
hcp_70()