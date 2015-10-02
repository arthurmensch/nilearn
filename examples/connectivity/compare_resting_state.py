from __future__ import division

import glob
from os.path import join
import os
import time
import pickle
import datetime
import warnings
import itertools
from joblib import delayed, Parallel, Memory
import numpy as np
import pandas as pd

import matplotlib
import shutil
from sklearn.utils import gen_even_slices, check_array, check_random_state
from nilearn._utils import check_niimg

matplotlib.use('PDF')

from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn import datasets
from nilearn.decomposition import SparsePCA, DictLearning, CanICA
from nilearn.decomposition.base import DecompositionEstimator, MaskReducer
from nilearn_sandbox.plotting.pdf_plotting import plot_to_pdf
from nilearn_sandbox._utils.map_alignment import spatial_correlation,\
    align_many_to_one_nii, _align_one_to_one_flat, _spatial_correlation_flat

from nilearn_sandbox.plotting.papaya import papaya_viewer

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
    evolution = sorted(glob.glob(join(output, 'debug', 'components_*.nii.gz')),
                       compare)

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
    elif dataset == 'hcp_reduced':
        dataset = datasets.fetch_hcp_reduced(n_subjects=n_subjects,
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
                                    slice=this_slice,
                                    random_state=estimator.random_state))

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
    return output


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
    mask_reducer = MaskReducer(masker,
                               memory_level=2,
                               memory=cache_dir, mock=False,
                               in_memory=False,
                               compression_type=
                               compression_type,
                               reduction_ratio=
                               reduction_ratio,
                               temp_folder=temp_folder,
                               mem_name='concat',
                               n_jobs=n_jobs)
    mask_reducer.fit(data_filenames)

    data = mask_reducer.data_
    subject_limits = mask_reducer.subject_limits_
    # data = np.load(os.path.expanduser('~/temp/hcp40_025_subsample.npy'),
    #                                   mmap_mode='r')
    # subject_limits = np.arange(0, 154 * 300, 300)
    print(data.shape)
    print(subject_limits)

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


def display_stability(output, ref_indices, target_indices):
    ref_components_list = []
    target_components_list = []
    n_batch = len(target_indices)
    full_n_exp = len(ref_indices)
    n_run = 3
    n_exp = full_n_exp // n_run

    masker = MultiNiftiMasker(mask_img=join(output, 'mask_img.nii.gz')).fit()

    try:
        os.makedirs(join(output, 'stability'))
    except:
        pass
    stability_output = join(output, 'stability')

    for i in ref_indices:
        components = masker.transform(join(output, 'comparison',
                                           'experiment_%i.nii.gz' % i))
        ref_components_list.append(components)
    for k in range(n_batch):
        target_components_list.append([])
        for i in target_indices[k]:
            components = masker.transform(join(output, 'comparison',
                                               'experiment_%i.nii.gz' % i))
            target_components_list[k].append(components)
    corr = np.zeros((n_batch, n_exp, n_run))
    mem = Memory(cachedir='nilearn_cache')
    labels = ['Full vs Full', 'Full vs Subsample', 'Full vs Rangefinder']
    plt.figure()
    for k in range(n_batch):
        base = np.concatenate(ref_components_list)
        target = np.concatenate(target_components_list[k])
        aligned = mem.cache(_align_one_to_one_flat)(base, target)
        diag = (_spatial_correlation_flat(aligned, base)).diagonal()
        plt.hist(diag, bins=60, label=labels[k])
        # plt.plot(np.linspace(0, 1, 600), np.cumsum(np.sort(diag)) /
        #          (np.arange(600) + 1), label=labels[k])
    plt.xlabel('Ratio of worst aligned components')
    plt.ylabel('Mean correlation')
    plt.legend()
    plt.savefig('figure.pdf')
    return
    for k in range(n_batch):
        for j in range(n_run):
            random_state = check_random_state(j)
            random_state.shuffle(target_components_list[k])
            random_state.shuffle(ref_components_list)
            for i in range(0, n_exp):
                base = np.concatenate(ref_components_list[j*n_exp:j*n_exp+i+1])
                target = np.concatenate(target_components_list[k][j*n_exp:j*n_exp+i+1])
                aligned = mem.cache(_align_one_to_one_flat)(base, target)
                corr[k, i, j] = np.trace(_spatial_correlation_flat(aligned,
                                                                   base))
                corr[k, i, j] /= base.shape[0]
                print(i, j)

    color = ['blue', 'green', 'red', 'yellow']
    plt.figure()
    for k in range(n_batch):
        plt.plot(np.arange(corr.shape[1])+1, np.mean(corr[k], axis=1), color=color[k],
                 label=k)
        plt.fill_between(np.arange(corr.shape[1])+1, np.min(corr[k], axis=1), np.max(corr[k],
                                                                     axis=1),
                         facecolor=color[k], alpha=0.3)
    plt.legend()
    plt.xlabel('# experiments')
    plt.ylabel('Map recovery')
    plt.savefig(join(stability_output, 'inc_correlation.pdf'))
    for k in range(n_batch):
        base = np.concatenate(ref_components_list)
        target = np.concatenate(target_components_list[k])
        aligned = _align_one_to_one_flat(base, target)
        corr = _spatial_correlation_flat(aligned, base)
        index = np.argsort(corr.diagonal())[::-1]

        base = base[index]
        aligned = aligned[index]

        corr = _spatial_correlation_flat(aligned, base)
        plt.figure()
        plt.matshow(corr)
        plt.savefig(join(stability_output, 'mat_corr_%i.pdf' % k))
        masker.inverse_transform(aligned).to_filename(join(stability_output, 'aligned_%i.nii.gz' % k))
        masker.inverse_transform(base).to_filename(join(stability_output, 'base_%i.nii.gz' % k))
        papaya_viewer(join(stability_output, 'aligned_%i.nii.gz' % k), output_file=join(stability_output, 'aligned_%i.html' % k))
        papaya_viewer(join(stability_output, 'base_%i.nii.gz' % k), output_file=join(stability_output, 'base_%i.html' % k))


def display_figures(output):
    results = pickle.load(open(join(output, 'results'), 'rb'))
    df = pd.DataFrame(results, columns=results[0].keys())
    df['reduction_ratio'] = np.round(df['reduction_ratio'], decimals=3)
    df['index_col'] = df.index
    df = df.set_index(['compression_type', 'reduction_ratio',
                       'alpha', 'random_state'])
    df_plot = df.loc[df.groupby(level=['compression_type',
                                'reduction_ratio'])[
        'score'].idxmax()][['score', 'math_time', 'index_col']]
    df_plot.reset_index(level=2, inplace=True)
    df_plot = df_plot.mean(level=['compression_type', 'reduction_ratio'])
    max_time = df_plot['math_time'][-1]
    plt.figure()

    for compression_type in ['range_finder', 'subsample']:
        this_df = df_plot.loc[compression_type].sort('math_time')
        plt.plot(this_df['math_time'] / max_time, this_df['score'], '-',
                 marker='o', label=compression_type)
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), '--', color='black')
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Time Gain')
    plt.xlim([0, 8])
    plt.ylim([0, 1])
    plt.savefig(join(output, 'score_vs_time.pdf'))

    plt.figure()
    for compression_type in ['range_finder', 'subsample']:
        this_df = df_plot.loc[compression_type]
        plt.plot(this_df['math_time'].index.values,
                 this_df['math_time'], '-',
                 marker='o', label=compression_type)
    plt.legend()
    plt.ylabel('Time')
    plt.xlabel('Reduction ratio')
    plt.savefig(join(output, 'time.pdf'))

    plt.figure()
    for compression_type in ['range_finder', 'subsample']:
        this_df = df_plot.loc[compression_type]
        plt.plot(this_df.index.values,
                 this_df['score'], '-',
                 marker='o', label=compression_type)
    for x in this_df.index.values:
        plt.annotate('      %i' % this_df.loc[x]['index_col'],
                     xy=(x, this_df.loc[x]['score']))
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Reduction ratio')
    plt.savefig(join(output, 'score.pdf'))


if __name__ == '__main__':
    # display_figures('/volatile/arthur/work/output/compare/2015-09-29_09-12-51')
    exit(0)
    # for compression_type in ['range_finder', 'subsample']:
    #     for reduction_ratio in np.linspace(0.1, 1, 10):
    #         for alpha in np.linspace(2, 20, 10):
    #             for parity in [0, 1]:
    #                 estimators.append(DictLearning(alpha=alpha, batch_size=20,
    #                                                compression_type=
    #                                                compression_type,
    #                                                random_state=0,
    #                                                forget_rate=1,
    #                                                reduction_ratio=reduction_ratio,
    #                                                in_memory=True,
    #                                                parity=parity))
    # reference = (np.arange(len(estimators)) // 2 * 2) + 1
    # run_experiment(estimators, n_split=1, n_jobs=20, dataset='adhd',
    # #                n_subjects=40,
    # #                smoothing_fwhm=6.,
    # #                init="rsn20",
    # #                n_epochs=1,
    # #                reference=reference)
    t0 = time.time()
    estimators = []
    # try:
    #     shutil.rmtree(os.path.expanduser('~/nilearn_cache/joblib/sklearn'))
    # except:
    #     pass
    # for compression_type in ['range_finder', 'subsample']:
    #     for reduction_ratio in np.linspace(0.1, 1, 10):
    #         for alpha in np.linspace(2, 20, 10):
    #             estimators.append(DictLearning(alpha=alpha, batch_size=20,
    #                                            compression_type=
    #                                            compression_type,
    #                                            random_state=0,
    #                                            forget_rate=1,
    #                                            reduction_ratio=reduction_ratio,
    #                                            in_memory=True))
    # estimators.append(DictLearning(alpha=10, batch_size=20,
    #                                compression_type='none',
    #                                random_state=0,
    #                                forget_rate=1,
    #                                reduction_ratio=1.,
    #                                in_memory=True))
    # reference = np.ones(len(estimators), dtype='int') * (len(estimators) - 1)
    # run_experiment(estimators, n_split=1, n_jobs=24, dataset='adhd',
    #                n_subjects=40,
    #                smoothing_fwhm=6.,
    #                init=os.path.expanduser('~/ica/'
    #                                        'canica_resting_state_20.nii.gz'),
    #                n_epochs=1,
    #                reference=reference)
    # estimators = []
    # for compression_type in ['range_finder', 'subsample']:
    #     for reduction_ratio in np.linspace(0.1, 1, 10):
    #                 estimators.append(SparsePCA(alpha=0.1, batch_size=20,
    #                                             compression_type=
    #                                             compression_type,
    #                                             random_state=0,
    #                                             forget_rate=1,
    #                                             reduction_ratio=reduction_ratio))
    # for random_state in range(0, 40):
    #     estimators.append(DictLearning(alpha=20, batch_size=20,
    #                                    compression_type=
    #                                    'none',
    #                                    random_state=random_state,
    #                                    forget_rate=1,
    #                                    reduction_ratio=1))
    # estimators.append(DictLearning(alpha=20, batch_size=20,
    #                                compression_type=
    #                                'none',
    #                                random_state=random_state,
    #                                forget_rate=1,
    #                                reduction_ratio=1))
    # estimators.append(SparsePCA(alpha=0.1, batch_size=20,
    #                             compression_type=
    #                             'none',
    #                             random_state=random_state,
    #                             forget_rate=1,
    #                             reduction_ratio=1))
    # reference = np.ones(len(estimators), dtype='int') * (len(estimators) - 1)
    # #
    # #
    # #
    # # n_exp = 10
    # # n_run = 1
    # # for random_state in range(0, n_exp * n_run):
    # #     estimators.append(DictLearning(alpha=20, batch_size=20,
    # #                                 compression_type=
    # #                                 'none',
    # #                                 random_state=random_state,
    # #                                 forget_rate=1,
    # #                                 reduction_ratio=1))
    # # for random_state in range(n_exp * n_run, 2 * n_exp * n_run):
    # #     estimators.append(DictLearning(alpha=20, batch_size=20,
    # #                                 compression_type=
    # #                                 'none',
    # #                                 random_state=random_state,
    # #                                 forget_rate=1,
    # #                                 reduction_ratio=1))
    # # for random_state in range(n_exp * n_run, 2 * n_exp * n_run):
    # #     estimators.append(DictLearning(alpha=12, batch_size=20,
    # #                                 compression_type=
    # #                                 'subsample',
    # #                                 random_state=random_state,
    # #                                 forget_rate=1,
    # #                                 reduction_ratio=0.2))
    # # for random_state in range(n_exp * n_run, 2 * n_exp * n_run):
    # #     estimators.append(DictLearning(alpha=20, batch_size=20,
    # #                                 compression_type=
    # #                                 'range_finder',
    # #                                 random_state=random_state,
    # #                                 forget_rate=1,
    # #                                 reduction_ratio=0.2))
    # # reference = np.ones(len(estimators), dtype='int') * (len(estimators) - 1)
    # # output = run_experiment(estimators, n_split=1, n_jobs=15, dataset='adhd',
    # #                n_subjects=40,
    # #                smoothing_fwhm=6.,
    # #                init=os.path.expanduser('~/ica/canica_resting_state_20.nii.gz'),
    # #                n_epochs=1,
    # #                reference=reference)
    # # print(output)
    # display_stability('/volatile/arthur/drago_output/2015-09-30_16-12-07',
    #                   np.arange(30), [np.arange(30, 60), np.arange(60, 90),
    #                                   np.arange(90, 120)])
    # for alpha in [5, 10, 15, 20]:
    #     estimators.append(DictLearning(alpha=alpha, batch_size=20,
    #                                    random_state=0))
    # reference = np.ones(len(estimators), dtype='int') * (len(estimators) - 1)
    # run_experiment(estimators, n_split=1, init='rsn70',
    #                n_epochs=1,
    #                dataset='hcp_reduced',
    #                n_subjects=4,
    #                smoothing_fwhm=6.,
    #                n_jobs=4, parallel_exp=True,
    #                reference=reference)
    # time = time.time() - t0
    # print('Total_time : %f s' % time)
