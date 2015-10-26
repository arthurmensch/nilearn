from os.path import join, expanduser
import numpy as np

from nilearn.decomposition import DictLearning
from nilearn.decomposition.isbi_utils import run, \
    analyse, analyse_num_exp, plot_num_exp, plot_full, \
    drop_memmmap, Experiment, convert_nii_to_pdf, \
    clean_memory, gather_results, analyse_median_maps, plot_median, \
    plot_full_multiple


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
                            reference=False,
                            temp_folder=expanduser('~/temp'),
                            n_runs=1)

    # temp_folder = drop_memmmap(estimators, experiment)
    # temp_folder = '/volatile/arthur/temp/2015-10-15_17-38-44'
    # output_dir = run(estimators, experiment)
    output_dir = expanduser('~/output/2015-10-14_21-02-57')
    # analyse_median_maps(output_dir)
    # gather_results(output_dir=output_dir)
    # analyse(output_dir, n_jobs=28, limit=12)
    # analyse_num_exp(output_dir, n_jobs=20, n_run_var=4)
    # analyse_median_maps(output_dir)
    # plot_num_exp(output_dir)
    # plot_full(output_dir, n_exp=9)
    plot_num_exp(output_dir, reduction_ratio_list=[0.05, 0.2], n_exp=9)
    # plot_full(output_dir)
    # plot_median(output_dir)
    # plot_num_exp(output_dir, 0.1)
    # convert_nii_to_pdf(join(output_dir, 'stability'), n_jobs=15)

def plot_full():
    output_dir_list = expanduser('~/output/2015-10-14_21-02-57'),\
                 expanduser('~/output/2015-10-14_23-46-52')
    n_exp_list = 9, 2
    plot_full_multiple(output_dir_list, n_exp_list)



def hcp_70():
    # HCP RSN70 explorative experiment
    estimators = []
    # alpha_list = np.array([[5, 6, 5, 5, 4, 4, 4, 4, 4, 4],
    #                        [2, 3, 3, 3, 4, 3, 3, 4, 4, 4]])

    # estimators.append(DictLearning(alpha=6, batch_size=20,
    #                                compression_type='subsample',
    #                                random_state=0,
    #                                forget_rate=1,
    #                                reduction_ratio=1))
    for compression_type in ['subsample']:
        for reduction_ratio, alpha in ((0.8, 6), (.9, 6), (1, 6)):
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
                            n_jobs=30,
                            n_epochs=1,
                            reference=False,
                            # Out of core dictionary learning specifics
                            temp_folder=expanduser('~/temp'),
                            # Stability specific
                            n_runs=10)
    # temp_folder = '/home/parietal/amensch/temp/2015-10-12_17-06-34'
    # temp_folder = '/home/parietal/amensch/temp/2015-10-20_08-19-37'
    # temp_folder = drop_memmmap(estimators, experiment)
    # output_dir = run(estimators, experiment, temp_folder=temp_folder)
    output_dir = expanduser('~/output/2015-10-14_23-46-52')
    # output_dir = expanduser('~/output/2015-10-23_14-03-53')
    # gather_results(output_dir)
    # analyse(experiment, output_dir, n_jobs=5, limit=9)
    analyse_median_maps(output_dir, reduction_ratio=0.025)
    plot_median(output_dir)
    # analyse_num_exp(experiment, output_dir, n_jobs=5, limit=9,
    #                 n_run_var=3)
    # plot_full(output_dir, n_exp=2)
    # plot_num_exp(output_dir, reduction_ratio_list=[0.025, 0.2], n_exp=2)
    # plot_full(output_dir)


def hcp_intensive():
    # HCP RSN70 explorative experiment
    # estimators = []
    # alpha_list = np.array([[6, 6, 5, 6],
    #                        [2, 2, 2, 2]])
    # experiment = Experiment('hcp_reduced',
    #                         n_subjects=75,
    #                         smoothing_fwhm=6,
    #                         dict_init='rsn70',
    #                         output_dir=expanduser('~/output'),
    #                         cachedir=expanduser('~/nilearn_cache'),
    #                         data_dir=expanduser('~/data'),
    #                         n_slices=1,
    #                         n_jobs=32,
    #                         n_epochs=1,
    #                         reference=False,
    #                         # Out of core dictionary learning specifics
    #                         temp_folder=expanduser('~/temp'),
    #                         # Stability specific
    #                         n_runs=20)
    # for i, compression_type in enumerate(['range_finder', 'subsample']):
    #     for reduction_ratio, alpha in zip([0.01, 0.025, 0.05, 0.075, 0.1, 0.2,
    #                                        0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    #                                        1.0],
    #                                       alpha_list[i]):
    #         estimators.append(DictLearning(alpha=alpha, batch_size=20,
    #                                        compression_type=compression_type,
    #                                        random_state=0,
    #                                        forget_rate=1,
    #                                        reduction_ratio=reduction_ratio))
    # # temp_folder = '/home/parietal/amensch/temp/2015-10-12_17-06-34'
    # temp_folder = '/home/parietal/amensch/temp/2015-10-20_08-19-37'
    # # temp_folder = drop_memmmap(estimators, experiment)
    # output_dir = run(estimators, experiment, temp_folder=temp_folder)
    #
    estimators = []
    alpha_list = [[7, 6, 6, 5, 6, 6, 7, 7, 6, 6],
                  [2, 4, 4, 3, 4, 4, 4, 6, 6, 6]]
    experiment = Experiment('hcp_reduced',
                            n_subjects=75,
                            smoothing_fwhm=6,
                            dict_init='rsn70',
                            output_dir=expanduser('~/output'),
                            cachedir=expanduser('~/nilearn_cache'),
                            data_dir=expanduser('~/data'),
                            n_slices=1,
                            n_jobs=32,
                            n_epochs=1,
                            reference=True,
                            # Out of core dictionary learning specifics
                            temp_folder=expanduser('~/temp'),
                            # Stability specific
                            n_runs=20)

    estimators.append(DictLearning(alpha=6, batch_size=20,
                                   compression_type='subsample',
                                   random_state=0,
                                   forget_rate=1,
                                   reduction_ratio=1))
    for i, compression_type in enumerate(['range_finder', 'subsample']):
        for reduction_ratio, alpha in zip(np.linspace(0.1, 1, 10),
                                          alpha_list[i]):
            estimators.append(DictLearning(alpha=alpha, batch_size=20,
                                           compression_type=compression_type,
                                           random_state=0,
                                           forget_rate=1,
                                           reduction_ratio=reduction_ratio))

    # temp_folder = '/home/parietal/amensch/temp/2015-10-12_17-06-34'
    # output_dir = run(estimators, experiment, temp_folder=temp_folder)
    output_dir = expanduser('~/output/2015-10-14_23-46-52')
    # gather_results(output_dir)
    # analyse(experiment, output_dir, n_jobs=32, limit=9)
    # analyse_num_exp(experiment, output_dir, n_jobs=32, limit=9,
    #                 n_run_var=3)
    # plot_full(output_dir, n_exp=2)
    # plot_num_exp(output_dir, reduction_ratio_list=[0.025, 0.2], n_exp=2)
    # plot_full(output_dir)


def hcp_full_70():
    # HCP RSN70 explorative experiment
    estimators = []
    for alpha in np.linspace(16, 32, 16):
        estimators.append(DictLearning(alpha=alpha, batch_size=20,
                                       compression_type='range_finder',
                                       random_state=0,
                                       forget_rate=1,
                                       reduction_ratio=0.05))
    experiment = Experiment('hcp',
                            n_subjects=100,
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
                            n_runs=1)
    # temp_folder = expanduser('~/temp/2015-10-15_23-45-45')
    # temp_folder = drop_memmmap(estimators, experiment)
    # temp_folder = expanduser('~/temp/2015-10-20_23-54-30')
    # output_dir = run(estimators, experiment, temp_folder=temp_folder)
    output_dir = expanduser('~/output/2015-10-14_23-46-52')
    gather_results(output_dir)
    analyse(experiment, output_dir, n_jobs=20, limit=1)
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
# hcp_intensive()
hcp_70()
# plot_full()