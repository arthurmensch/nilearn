from __future__ import division

import glob
from os.path import join
import os
import time
import pickle
import datetime
import warnings
import itertools
from joblib import delayed, Parallel
import numpy as np
import pandas as pd

import matplotlib
from sklearn.utils import gen_even_slices
from nilearn._utils import check_niimg

matplotlib.use('PDF')

from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn import datasets
from nilearn.decomposition import SparsePCA, DictLearning, CanICA
from nilearn.decomposition.base import DecompositionEstimator, MaskReducer
from nilearn_sandbox.plotting.pdf_plotting import plot_to_pdf
from nilearn_sandbox._utils.map_alignment import spatial_correlation, align_many_to_one_nii

import matplotlib.pyplot as plt


def compare(x, y):
    if len(x) < len(y):
        return -1
    elif len(x) == len(y):
        return cmp(x, y)
    else:
        return 1


def dump_single_experiment_debug(estimator, output):
    print('[Example] Dropping debug information')
    a4_size = (8.27, 11.69)
    size = len(estimator.debug_info_['residuals'])
    fig, axes = plt.subplots(3, 1, figsize=a4_size, sharex=True)
    titles = {'residuals': 'Residuals',
              'density': 'Density',
              'values': 'Voxels trajectories'}

    ylabels = {'residuals': 'Surrogate',
               'density': 'Density',
               'values': 'Voxel value'}
    for i, (key, ax) in enumerate(zip(sorted(estimator.debug_info_), axes)):
        ax.plot(np.array(estimator.debug_info_[key]))
        ax.set_xlim(0, size)
        ax.set_title(titles[key])
        if i == 2:
            ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabels[key])
    plt.savefig(join(output, 'debug.pdf'))
    plt.close(fig)
    with open(join(output, 'debug.txt'), 'w+') as f:
        if hasattr(estimator, 'score_'):
            f.write('Component score :')
            f.write(str(estimator.score_))
            f.write('\n')
        f.write('Timings :')
        f.write("Math %.2f - IO %.2f" % (estimator.time_[0], estimator.time_[1]))
        f.write('\n')
    # evolution = sorted(glob.glob(join(output, 'debug', 'components_*.nii.gz')),
    #                    compare)
    # with PdfPages(join(output, 'evolution.pdf')) as pdf:
    #     for i in range(0, len(evolution), 5):
    #         fig, axes = plt.subplots(5, 1, figsize=a4_size, squeeze=False)
    #         axes = axes.reshape(-1)
    #         for j, ax in enumerate(axes):
    #             if i + j < len(evolution):
    #                 plot_prob_atlas(evolution[i + j], axes=ax)
    #             else:
    #                 ax.axis('off')
    #         pdf.savefig(fig)
    #         plt.close()
    #
    # with PdfPages(join(output, 'evolution_single.pdf')) as pdf:
    #     for i in range(0, len(evolution), 5):
    #         fig, axes = plt.subplots(5, 1, figsize=a4_size, squeeze=False)
    #         axes = axes.reshape(-1)
    #         for j, ax in enumerate(axes):
    #             if i + j < len(evolution):
    #                 plot_stat_map(index_img(evolution[i + j], 0), axes=ax)
    #             else:
    #                 ax.axis('off')
    #         pdf.savefig(fig)
    #         plt.close()


def run_single_experiment(index, estimator, func_filenames, output):
    exp_output = join(output, "experiment_%i" % index)
    os.mkdir(exp_output)
    if type(estimator).__name__:
        debug_folder = join(exp_output, 'debug')
        os.mkdir(debug_folder)
    estimator.debug_folder = debug_folder
    print('[Example] Learning maps using %s model' % type(estimator).__name__)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        estimator.fit(func_filenames)
    full_time = time.time() - t0
    print('[Example] Dumping results')
    dump_single_experiment_debug(estimator, exp_output)
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_filename = join(exp_output, 'components.nii.gz')
    components_img.to_filename(components_filename)
    print('[Example] Preparing pdf')
    # plot_to_pdf(components_img, path=join(exp_output, 'components.pdf'))
    timing = np.zeros(3)
    timing[0:2] = estimator.time_
    timing[2] = full_time
    return components_filename, timing


def run_raw_single_experiment(index, estimator, data, output):
    exp_output = join(output, "experiment_%i" % index)
    os.mkdir(exp_output)
    if type(estimator).__name__:
        debug_folder = join(exp_output, 'debug')
        os.mkdir(debug_folder)
    estimator.debug_folder = debug_folder
    print('[Example] Learning maps using %s model' % type(estimator).__name__)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        estimator._raw_fit(data)
    full_time = time.time() - t0
    print('[Example] Dumping results')
    dump_single_experiment_debug(estimator, exp_output)
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_filename = join(exp_output, 'components.nii.gz')
    components_img.to_filename(components_filename)
    # print('[Example] Preparing pdf')
    # plot_to_pdf(components_img, path=join(exp_output, 'components.pdf'))
    timing = np.zeros(3)
    timing[0:2] = estimator.time_
    timing[2] = full_time
    return components_filename, timing


def dump_comparison(i, masker, components, reference, dump_dir):
    print("[Example] Dropping aligned components % i" % i)
    filename = join(dump_dir, "experiment_%i.nii.gz" % i)
    components.to_filename(filename)
    print('[Example] Preparing pdf %i' % i)
    plot_to_pdf(components, path=join(dump_dir,
                                    "experiment_%i.pdf" % i))
    corr = spatial_correlation(masker, components, reference)
    plt.matshow(corr, vmin=-1, vmax=1)
    plt.xlabel('Components')
    plt.ylabel('References')
    plt.savefig(join(dump_dir, 'corr_%i.pdf' % i))
    return np.diagonal(corr), np.mean(np.diag(corr))


def run_experiment(estimators, n_split=1, init='rsn70', n_epochs=1,
                   dataset='adhd',
                   n_subjects=40,
                   smoothing_fwhm=4.,
                   n_jobs=6, parallel_exp=True,
                   reference=None):
    output = os.path.expanduser('~/output/compare')
    temp_folder = os.path.expanduser('~/temp')
    cache_dir = os.path.expanduser('~/nilearn_cache')
    data_dir = os.path.expanduser('~/data')
    output = join(output, datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                          '-%M-%S'))
    try:
        os.makedirs(output)
    except:
        pass

    if dataset == 'adhd':
        dataset = datasets.fetch_adhd(n_subjects=min(40, n_subjects))
        mask = os.path.expanduser('~/data/ADHD_mask/mask_img.nii.gz')
    elif dataset == 'hcp':
        dataset = datasets.fetch_hcp_rest(n_subjects=n_subjects,
                                          data_dir=data_dir)
        mask = os.path.expanduser('~/data/HCP_mask/mask_img.nii.gz')
    smith = datasets.fetch_atlas_smith_2009()
    if isinstance(init, int):
        dict_init = None
        n_components = init
    elif init == 'rsn70':
        dict_init = smith.rsn70
        n_components = 70
    elif init == 'rsn20':
        dict_init = smith.rsn20
        n_components = 20
    else:
        if os.path.exists(init):
            dict_init = init
            n_components = check_niimg(init).get_shape()[3]
        else:
            raise ValueError('Unsupported init')
    data_filenames = dataset.func

    print('First functional nifti image (4D) is at: %s' %
          dataset.func[0])

    slices = list(gen_even_slices(len(data_filenames), n_split))
    result_dict = []

    # For alignment
    if reference is None:
        reference = np.ones((len(estimators) * len(slices)), dtype='int')
        for i in range(len(reference)):
            reference[i] = (i // len(estimators)) * len(estimators)
    print(reference)

    # This is hacky and should be integrated in the nilearn API in a smooth way
    # Warming up cache with masked images
    print("[Example] Warming up cache")
    decomposition_estimator = DecompositionEstimator(smoothing_fwhm=
                                                     smoothing_fwhm,
                                                     memory=cache_dir,
                                                     mask=mask,
                                                     memory_level=2,
                                                     verbose=10,
                                                     n_jobs=n_jobs)
    decomposition_estimator.fit(data_filenames)
    masker = decomposition_estimator.masker_


    print("[Example] Warming up cache")
    mask_reducer = MaskReducer(masker,
                               memory_level=2,
                               memory=cache_dir, mock=True,
                               in_memory=True,
                               temp_folder=temp_folder,
                               n_jobs=n_jobs)
    mask_reducer.fit(data_filenames)

    estimator_n_jobs = n_jobs if not parallel_exp else 1

    for estimator in estimators:
        # Setting technical parameters
        estimator.set_params(mask=masker, dict_init=dict_init,
                             smoothing_fwhm=smoothing_fwhm,
                             n_components=n_components,
                             n_epochs=n_epochs,
                             n_jobs=estimator_n_jobs,
                             memory_level=2, memory='nilearn_cache',
                             verbose=3)
    with open(join(output, 'estimators'), 'w+') as f:
        for i, (estimator,
                this_slice) in enumerate(itertools.product(estimators,
                                                           slices)):
            f.write("%s\n" % estimator)
            result_dict.append(dict(type=type(estimator).__name__,
                                    alpha=estimator.alpha,
                                    batch_size=estimator.batch_size,
                                    compression_type=
                                    estimator.compression_type,
                                    reduction_ratio=
                                    estimator.reduction_ratio,
                                    power_iter=estimator.power_iter,
                                    # feature_compression=estimator.
                                    # feature_compression,
                                    forget_rate=estimator.forget_rate,
                                    slice=this_slice))

    exp_n_jobs = n_jobs if parallel_exp else 1

    res = Parallel(n_jobs=exp_n_jobs, verbose=10)\
        (delayed(run_single_experiment)(i,
                               estimator,
                               data_filenames[this_slice],
                               output)
         for i, (estimator,
                 this_slice) in enumerate(
            itertools.product(estimators, slices)))

    components_filename, timings_list = zip(*res)
    timings = np.zeros((len(estimators) * len(slices), 3))
    for i, timing in enumerate(timings_list):
        timings[i] = np.array(timing)
        result_dict[i]['math_time'] = timings[i, 0]
        result_dict[i]['io_time'] = timings[i, 1]
        result_dict[i]['total_time'] = timings[i, 2]
    np.save(join(output, 'timings'), timings)

    masker.mask_img_.to_filename(join(output, 'mask_img.nii.gz'))

    if len(slices) * len(estimators) > 1:
        print("Performing alignment")
        map_masker = MultiNiftiMasker(mask_img=masker.mask_img_,).fit()
        components_list = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(align_many_to_one_nii)(map_masker,
                                           components_filename[
                                           reference[i]],
                                           target_components)
            for i, target_components in enumerate(components_filename))
        comparison_dir = join(output, "comparison")
        os.mkdir(comparison_dir)
        res = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(dump_comparison)(i, map_masker, components,
                                     components_filename[reference[i]],
                                     comparison_dir)
            for i, components in enumerate(components_list))
        diag_list, score_list = zip(*res)
        np.save('correlations', np.array(score_list))
        np.save('diag', np.concatenate([this_diag[np.newaxis, :]
                                        for this_diag in diag_list]))
        for i, (diag, score) in enumerate(zip(diag_list, score_list)):
            result_dict[i]['score'] = score
            result_dict[i]['diag'] = diag_list[i]
    pickle.dump(result_dict, open(join(output, 'results'), 'w+'))
    with open(join(output, 'results.txt'), 'w+') as f:
        for exp_dict in result_dict:
            f.write("%s\n" % exp_dict)
    # display_figures(output)


def run_dict_learning_experiment(estimators, n_split=1, init='rsn70', n_epochs=1,
                                 compression_type='subsample',
                                 dataset='adhd',
                                 reduction_ratio=1.,
                                 n_subjects=40,
                                 smoothing_fwhm=4.,
                                 n_jobs=6, parallel_exp=True,
                                 reference=None):
    output = os.path.expanduser('~/output/compare')
    temp_folder = os.path.expanduser('~/temp')
    cache_dir = os.path.expanduser('~/nilearn_cache')
    data_dir = os.path.expanduser('~/data')
    output = join(output, datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                           '-%M-%S'))
    try:
        os.makedirs(output)
    except:
        pass

    if dataset == 'adhd':
        dataset = datasets.fetch_adhd(n_subjects=min(40, n_subjects))
        mask = os.path.expanduser('~/data/ADHD_mask/mask_img.nii.gz')
    elif dataset == 'hcp':
        dataset = datasets.fetch_hcp_rest(n_subjects=n_subjects,
                                          data_dir=data_dir)
        mask = os.path.expanduser('~/data/HCP_mask/mask_img.nii.gz')
    smith = datasets.fetch_atlas_smith_2009()
    if isinstance(init, int):
        dict_init = None
        n_components = init
    elif init == 'rsn70':
        dict_init = smith.rsn70
        n_components = 70
    elif init == 'rsn20':
        dict_init = smith.rsn20
        n_components = 20
    else:
        if os.path.exists(init):
            dict_init = init
            n_components = check_niimg(init).get_shape()[3]
        else:
            raise ValueError('Unsupported init')
    data_filenames = dataset.func

    print('First functional nifti image (4D) is at: %s' %
          dataset.func[0])

    slices = list(gen_even_slices(len(data_filenames), n_split))
    result_dict = []

    # For alignment
    if reference is None:
        reference = np.ones((len(estimators) * len(slices)), dtype='int')
        for i in range(len(reference)):
            reference[i] = (i // len(estimators)) * len(estimators)
    print(reference)

    # This is hacky and should be integrated in the nilearn API in a smooth way
    # Warming up cache with masked images
    decomposition_estimator = DecompositionEstimator(smoothing_fwhm=
                                                     smoothing_fwhm,
                                                     memory=cache_dir,
                                                     mask=mask,
                                                     memory_level=2,
                                                     verbose=10,
                                                     n_jobs=n_jobs)
    decomposition_estimator.fit(data_filenames)
    masker = decomposition_estimator.masker_

    print("[Example] Warming up cache")
    # mask_reducer = MaskReducer(masker,
    #                            memory_level=2,
    #                            memory=cache_dir, mock=False,
    #                            in_memory=False,
    #                            compression_type=
    #                            compression_type,
    #                            reduction_ratio=
    #                            reduction_ratio,
    #                            temp_folder=temp_folder,
    #                            mem_name='concat',
    #                            n_jobs=n_jobs)
    # mask_reducer.fit(data_filenames)

    data = np.load(os.path.expanduser('~/temp/025subsample.npy'),
                   mmap_mode='r')
    subject_limits = np.arange(0, 46500, 300)# mask_reducer.data_
    # subject_limits = mask_reducer.subject_limits_

    estimator_n_jobs = n_jobs if not parallel_exp else 1

    for estimator in estimators:
        # Setting technical parameters
        estimator.set_params(mask=masker, dict_init=dict_init,
                             smoothing_fwhm=smoothing_fwhm,
                             reduction_ratio=reduction_ratio,
                             n_components=n_components,
                             n_epochs=n_epochs,
                             n_jobs=estimator_n_jobs,
                             in_memory=False,
                             memory_level=2, memory='nilearn_cache',
                             verbose=3)
    with open(join(output, 'estimators'), 'w+') as f:
        for i, (estimator,
                this_slice) in enumerate(itertools.product(estimators,
                                                           slices)):
            f.write("%s\n" % estimator)
            result_dict.append(dict(type=type(estimator).__name__,
                                    alpha=estimator.alpha,
                                    batch_size=estimator.batch_size,
                                    compression_type=
                                    estimator.compression_type,
                                    reduction_ratio=
                                    estimator.reduction_ratio,
                                    power_iter=estimator.power_iter,
                                    forget_rate=estimator.forget_rate,
                                    slice=this_slice,
                                    parity=estimator.parity))

    exp_n_jobs = n_jobs if parallel_exp else 1

    res = Parallel(n_jobs=exp_n_jobs, verbose=10)(
        delayed(run_raw_single_experiment)(i,
                                           estimator,
                                           data[subject_limits[
                                               this_slice.start]:
                                               subject_limits[
                                                   this_slice.stop]],
                                           output)
         for i, (estimator,
                 this_slice) in enumerate(
            itertools.product(estimators, slices)))

    components_filename, timings_list = zip(*res)
    timings = np.zeros((len(estimators) * len(slices), 3))
    for i, timing in enumerate(timings_list):
        timings[i] = np.array(timing)
        result_dict[i]['math_time'] = timings[i, 0]
        result_dict[i]['io_time'] = timings[i, 1]
        result_dict[i]['total_time'] = timings[i, 2]
    np.save(join(output, 'timings'), timings)

    masker.mask_img_.to_filename(join(output, 'mask_img.nii.gz'))

    if len(slices) * len(estimators) > 1:
        print("Performing alignment")
        map_masker = MultiNiftiMasker(mask_img=masker.mask_img_,).fit()
        components_list = []
        for i, target_components in enumerate(components_filename):
            components_list.append(align_many_to_one_nii(map_masker,
                                                    components_filename[
                                                        reference[i]],
                                                    target_components))
        comparison_dir = join(output, "comparison")
        os.mkdir(comparison_dir)
        corr_list = Parallel(n_jobs=n_jobs)(
            delayed(dump_comparison)(i, map_masker, components,
                                     components_filename[reference[i]],
                                     comparison_dir)
            for i, components in enumerate(components_list))
        np.save('correlations', np.array(corr_list))
        for i, score in enumerate(corr_list):
            result_dict[i]['score'] = score
    pickle.dump(result_dict, open(join(output, 'results'), 'w+'))
    with open(join(output, 'results.txt'), 'w+') as f:
        for exp_dict in result_dict:
            f.write("%s\n" % exp_dict)
    display_figures(output)


def display_figures(output):
    results = pickle.load(open(join(output, 'results'), 'rb'))
    df = pd.DataFrame(results, columns=results[0].keys())
    subdf = pd.DataFrame(columns=['compression_type', 'reduction_ratio',
                                  'io_time', 'math_time', 'total_time',
                                  'score', 'alpha'])
    for compression_type in ['range_finder', 'subsample']:
        for j, reduction_ratio in np.unique(df['reduction_ratio']):
            df_temp = df[np.logical_and(df[
                            'reduction_ratio'] ==
                                        reduction_ratio,
                         df['compression_type'] == compression_type)]
            idx = np.argmax(df_temp['score'])
            subdf = subdf.append(df.iloc[idx][['compression_type',
                                               'reduction_ratio',
                                               'io_time', 'math_time',
                                               'total_time', 'score',
                                               'alpha']])
    subsample_df = subdf[subdf['compression_type'] == 'subsample']
    rangefinder_df = subdf[subdf['compression_type'] == 'range_finder']

    plt.figure()
    plt.plot(subsample_df['reduction_ratio'], subsample_df['score'],
             marker='o', label='Subsample')
    plt.plot(rangefinder_df['reduction_ratio'], rangefinder_df['score'],
             marker='o', label='Range Finder')
    plt.legend(loc='upper left')
    plt.title('Correlation with baseline component maps (non compressed)')
    plt.xlabel('Compression rate')
    plt.ylabel('max_\Omega Tr(V_1 V_2^T \Omega)')
    plt.savefig(join(output, 'correlation.pdf'))

    plt.figure()

    plt.plot(subsample_df['reduction_ratio'], subsample_df['total_time'],
             color='b', marker = 'o', label='Subsample')
    plt.plot(rangefinder_df['reduction_ratio'], rangefinder_df['total_time'],
             color='r', marker='o', label='Range Finder')
    plt.legend(loc='upper left')
    plt.title('Correlation with baseline component maps (non compressed)')
    plt.ylabel('Time')
    plt.xlabel('Reduction ratio')
    plt.savefig(join(output, 'time.pdf'))

    plt.figure()

    max_score = np.max(subsample_df['score'])
    max_time = np.max(subsample_df['total_time'])
    plt.plot(subsample_df[['total_time']] / max_time,
             subsample_df['score'] / max_score, color='b',
             marker='o', label='Subsample')
    max_score = np.max(rangefinder_df['score'])
    max_time = np.max(rangefinder_df['total_time'])
    plt.plot(rangefinder_df[['total_time']] / max_time,
             rangefinder_df['score'] / max_score, color='r',
             marker='o', label='Range Finder')
    plt.plot(np.linspace(0.1, 1, 10), np.linspace(0.1, 1, 10),
             '--', color='black')
    plt.legend(loc='lower right')
    plt.title('Correlation with baseline component maps (non compressed)')
    plt.xlabel('Time')
    plt.ylabel('max_\Omega Tr(V_1 V_2^T \Omega)')
    plt.savefig(join(output, 'time_vs_correlation.pdf'))


if __name__ == '__main__':
    t0 = time.time()
    estimators = []
    # # # for compression_type in ['range_finder', 'subsample']:
    # # #     for reduction_ratio in np.linspace(0.1, 1, 10):
    # # #         for alpha in np.linspace(2, 20, 10):
    # # #             for parity in [0, 1]:
    # # #                 estimators.append(DictLearning(alpha=alpha, batch_size=20,
    # # #                                                compression_type=
    # # #                                                compression_type,
    # # #                                                random_state=0,
    # # #                                                forget_rate=1,
    # # #                                                reduction_ratio=reduction_ratio,
    # # #                                                in_memory=True,
    # # #                                                parity=parity))
    # # # reference = (np.arange(len(estimators)) // 2 * 2) + 1
    # # # run_experiment(estimators, n_split=1, n_jobs=20, dataset='adhd',
    # # #                n_subjects=40,
    # # #                smoothing_fwhm=6.,
    # # #                init="rsn20",
    # # #                n_epochs=1,
    # # #                reference=reference)
    # #
    # #
    # #
    # estimators = []
    # for compression_type in ['range_finder', 'subsample']:
    #     for reduction_ratio in np.linspace(0.1, 1, 10):
    #         for alpha in np.linspace(8, 20, 7):
    #             estimators.append(DictLearning(alpha=20, batch_size=20,
    #                                            compression_type=
    #                                            'none',
    #                                            random_state=0,
    #                                            forget_rate=1,
    #                                            reduction_ratio=1.,
    #                                            in_memory=True))
    # estimators.append(DictLearning(alpha=20, batch_size=20,
    #                            compression_type=
    #                            'none',
    #                            random_state=1234,
    #                            forget_rate=1,
    #                            reduction_ratio=1.,
    #                            in_memory=True))
    # reference = np.ones(len(estimators), dtype='int') * (len(estimators) - 1)
    # run_experiment(estimators, n_split=1, n_jobs=20, dataset='adhd',
    #                n_subjects=40,
    #                smoothing_fwhm=6.,
    #                init=os.path.expanduser('~/ica/canica_resting_state_20.nii.gz'),
    #                n_epochs=1,
    #                reference=reference)
    #
    # for compression_type in ['range_finder', 'subsample']:
    #     for reduction_ratio in np.linspace(0.1, 1, 10):
    #         for alpha in np.linspace(2, 12, 6):
    #             estimators.append(DictLearning(alpha=20, batch_size=20,
    #                                            compression_type=
    #                                            'none',
    #                                            random_state=0,
    #                                            forget_rate=1,
    #                                            reduction_ratio=1.,
    #                                            in_memory=True))
    # estimators.append(DictLearning(alpha=12, batch_size=20,
    #                            compression_type=
    #                            'none',
    #                            random_state=1234,
    #                            forget_rate=1,
    #                            reduction_ratio=1.,
    #                            in_memory=True))
    # reference = np.ones(len(estimators), dtype='int') * (len(estimators) - 1)
    # run_experiment(estimators, n_split=1, n_jobs=20, dataset='adhd',
    #                n_subjects=40,
    #                smoothing_fwhm=6.,
    #                init=os.path.expanduser('~/ica/canica_resting_state_70.nii.gz'),
    #                n_epochs=1,
    #                reference=reference)

    for alpha in [10, 20, 30, 40, 50]:
        estimators.append(DictLearning(alpha=alpha, batch_size=20,
                                       random_state=0))
    run_dict_learning_experiment(estimators, n_split=1, init='rsn70',
                                 n_epochs=1,
                                 dataset='hcp',
                                 reduction_ratio=0.25,
                                 compression_type='subsample',
                                 n_subjects=40,
                                 smoothing_fwhm=6.,
                                 n_jobs=6, parallel_exp=True,
                                 reference=None)
    time = time.time() - t0
    print('Total_time : %f s' % time)