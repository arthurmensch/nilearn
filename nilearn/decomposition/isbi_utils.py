import fnmatch

import collections
import glob
import os
from os.path import expanduser, join, exists

import datetime
import shutil
import warnings

import json
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.utils import gen_even_slices
from sklearn.base import clone

from nilearn_sandbox._utils.map_alignment import _align_one_to_one_flat, \
    _spatial_correlation_flat, spatial_correlation, align_many_to_one_nii

from . import SparsePCA, DictLearning
from .. import datasets
from .base import MaskReducer, DecompositionEstimator
from .._utils import check_niimg
from ..input_data import MultiNiftiMasker
from nilearn.image import index_img

Experiment = collections.namedtuple('Experiment',
                                    ['dataset_name',
                                     'n_subjects',
                                     'smoothing_fwhm',
                                     'dict_init',
                                     'output_dir',
                                     'cache_dir',
                                     'data_dir',
                                     'n_slices',
                                     'n_jobs',
                                     'n_epochs',
                                     'temp_folder',
                                     'n_runs'])


def load_dataset(exp_params, output_dir=None, warmup=True):
    n_subjects = exp_params.n_subjects
    data_dir = exp_params.data_dir

    if exp_params.dataset_name == 'adhd':
        dataset = datasets.fetch_adhd(n_subjects=
                                      min(40, n_subjects)).func
        mask = join(exp_params.data_dir, 'ADHD_mask', 'mask_img.nii.gz')
    elif exp_params.dataset_name == 'hcp':
        dataset = datasets.fetch_hcp_rest(n_subjects=n_subjects,
                                          data_dir=data_dir).func
        mask = join(exp_params.data_dir, 'HCP_mask', 'mask_img.nii.gz')
    elif exp_params.dataset_name == 'hcp_reduced':
        dataset = datasets.fetch_hcp_reduced(n_subjects=
                                             n_subjects,
                                             data_dir=data_dir).func
        mask = join(exp_params.data_dir, 'HCP_mask', 'mask_img.nii.gz')
    else:
        raise ValueError("Dataset not supported")
    if output_dir is not None:
        check_niimg(mask).to_filename(join(output_dir, 'mask_img.nii.gz'))

    print('[Experiment] Computing global mask')
    smoothing_fwhm = exp_params.smoothing_fwhm
    cache_dir = exp_params.cache_dir
    n_jobs = exp_params.n_jobs

    decomposition_estimator = DecompositionEstimator(smoothing_fwhm=
                                                     smoothing_fwhm,
                                                     memory=cache_dir,
                                                     mask=mask,
                                                     memory_level=2,
                                                     verbose=10,
                                                     n_jobs=n_jobs)
    decomposition_estimator.fit(dataset)

    masker = decomposition_estimator.masker_
    masker.mask_img_.get_data()

    if warmup:
        print("[Experiment] Warming up cache")
        mask_reducer = MaskReducer(masker,
                                   memory_level=2,
                                   memory=cache_dir,
                                   n_jobs=n_jobs,
                                   mock=True,
                                   in_memory=True
                                   )
        mask_reducer.fit(dataset)
    return dataset, masker


def check_init(exp_params):
    smith = datasets.fetch_atlas_smith_2009()
    if exp_params.dict_init == 'rsn70':
        dict_init = smith.rsn70
        n_components = 70
    elif exp_params.dict_init == 'rsn20':
        dict_init = smith.rsn20
        n_components = 20
    elif exists(exp_params.dict_init):
        dict_init = exp_params.dict_init
        n_components = check_niimg(exp_params.dict_init).get_shape()[3]
    else:
        raise ValueError('Init not supported')
    return dict_init, n_components


def yield_estimators(estimators, exp_params, masker, dict_init, n_components):
    smoothing_fwhm = exp_params.smoothing_fwhm
    n_epochs = exp_params.n_epochs
    n_runs = exp_params.n_runs
    cache_dir = exp_params.cache_dir
    for random_state in np.arange(n_runs):
        for i, estimator in enumerate(estimators):
            reference = (i == 0)
            offset = 100 if reference else 0
            estimator = clone(estimator)
            estimator.set_params(mask=masker,
                                 smoothing_fwhm=smoothing_fwhm,
                                 n_epochs=n_epochs,
                                 n_jobs=1,
                                 dict_init=dict_init,
                                 n_components=n_components,
                                 memory_level=2, memory=cache_dir,
                                 verbose=3,
                                 random_state=random_state + offset)
            yield estimator, reference


def single_run(index, estimator, dataset, output_dir, reference,
               data=None):
    exp_output = join(output_dir, "experiment_%i" % index)
    os.mkdir(exp_output)
    if type(estimator).__name__:
        debug_folder = join(exp_output, 'debug')
        os.mkdir(debug_folder)
    estimator.set_params(debug_folder=debug_folder)
    print('[Example] Learning maps')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if data is not None:
            memmap_data = np.memmap(data['filename'], dtype='float64',
                                    order='F', mode='r',
                                    shape=(
                                        data['n_samples'], data['n_voxels']))
            estimator._raw_fit(memmap_data)
            del memmap_data
        else:
            estimator.fit(dataset)
    print('[Example] Dumping results')
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_filename = join(exp_output, 'components.nii.gz')
    components_img.to_filename(components_filename)
    math_time = estimator.time_[0]
    io_time = estimator.time_[1]
    load_math_time = 0
    load_io_time = 0
    if data is not None:
        load_math_time += data['math_time']
        load_io_time += data['io_time']
    single_run_dict = {'estimator_type': type(estimator).__name__,
                       'compression_type': estimator.compression_type,
                       'reduction_ratio':
                           estimator.reduction_ratio,
                       'alpha': estimator.alpha,
                       'random_state': estimator.random_state,
                       # Columns
                       'components': components_filename,
                       'math_time': math_time,
                       'io_time': io_time,
                       'load_math_time': load_math_time,
                       'load_io_time': load_io_time,
                       'reference': reference
                       }
    with open(join(exp_output, 'results.json'), 'w+') as f:
        json.dump(single_run_dict, f)
    return single_run_dict


def run(estimators, exp_params, temp_folder=None):
    output_dir = join(exp_params.output_dir,
                      datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                       '-%M-%S'))
    os.mkdir(output_dir)
    with open(join(output_dir, 'experiment.json'), 'w+') as f:
        json.dump(exp_params.__dict__, f)

    dataset, masker = load_dataset(exp_params, output_dir=output_dir,
                                   warmup=(temp_folder is None))

    dataset_series = pd.Series(dataset)
    dataset_series.to_csv(join(output_dir, 'dataset.csv'))

    dict_init, n_components = check_init(exp_params)

    check_niimg(dict_init).to_filename(join(output_dir,
                                            'dict_init.nii.gz'))
    exp_estimators = list(yield_estimators(estimators,
                                           exp_params,
                                           masker,
                                           dict_init,
                                           n_components))
    if temp_folder is not None:
        estimators_data = pd.read_csv(join(temp_folder, 'data.csv'),
                                      index_col=0)
        estimators_data.reset_index(inplace=True)
        estimators_data.set_index(['compression_type', 'reduction_ratio'],
                                  inplace=True)
        estimators_data = [estimators_data.loc[estimator.compression_type,
                                               round(estimator.reduction_ratio,
                                                     3)]
                           for estimator, _ in exp_estimators]
    else:
        estimators_data = [None] * len(exp_estimators)
    full_dict_list = Parallel(n_jobs=exp_params.n_jobs)(
        delayed(single_run)(index, estimator, dataset, output_dir,
                            reference,
                            data=data)
        for index, ((estimator, reference), data) in
        enumerate(zip(exp_estimators, estimators_data)))
    results = pd.DataFrame(full_dict_list, columns=['estimator_type',
                                                    'compression_type',
                                                    'reduction_ratio',
                                                    'alpha',
                                                    'random_state',
                                                    'math_time', 'io_time',
                                                    'load_math_time',
                                                    'load_io_time',
                                                    'reference',
                                                    'components'])

    results.sort_index(by=['estimator_type',
                           'compression_type',
                           'reduction_ratio',
                           'alpha',
                           'random_state'], inplace=True)
    results.to_csv(join(output_dir, 'results.csv'))
    return output_dir


def gather_results(output_dir):
    full_dict_list = []
    for dirpath, dirname, filenames in os.walk(output_dir):
        for filename in fnmatch.filter(filenames, 'results.json'):
            with open(join(dirpath, filename), 'r') as f:
                full_dict_list.append(json.load(f))
    results = pd.DataFrame(full_dict_list, columns=['estimator_type',
                                                    'compression_type',
                                                    'reduction_ratio',
                                                    'alpha',
                                                    'random_state',
                                                    'math_time', 'io_time',
                                                    'load_math_time',
                                                    'load_io_time',
                                                    'reference',
                                                    'components'])

    results.sort_index(by=['estimator_type',
                           'compression_type',
                           'reduction_ratio',
                           'alpha',
                           'random_state'], inplace=True)
    results.to_csv(join(output_dir, 'results.csv'))


def single_drop_memmap(exp_params, temp_folder, index, dataset,
                       masker, compression_type, reduction_ratio):
    filename = 'experiment_%i' % index
    mask_reducer = MaskReducer(masker,
                               memory_level=2,
                               memory=exp_params.cache_dir,
                               n_jobs=1,
                               temp_folder=temp_folder,
                               filename=filename,
                               in_memory=False,
                               compression_type=compression_type,
                               reduction_ratio=reduction_ratio,
                               random_state=0)
    mask_reducer.fit(dataset)
    single_run_dict = {'compression_type': compression_type,
                       'reduction_ratio':
                           reduction_ratio,
                       'filename': join(temp_folder, filename),
                       'random_state': 0,
                       'math_time': mask_reducer.time_[0],
                       'io_time': mask_reducer.time_[1],
                       'n_samples': mask_reducer.data_.shape[0],
                       'n_voxels': mask_reducer.data_.shape[1]
                       }

    with open(join(temp_folder, filename + '.json'), 'w+') as f:
        json.dump(single_run_dict, f)
    return single_run_dict


def drop_memmmap(exp_params, estimators):
    base_temp_folder = exp_params.temp_folder
    temp_folder = join(base_temp_folder,
                       datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                        '-%M-%S'))
    try:
        os.mkdir(temp_folder)
    except:
        pass
    with open(join(temp_folder, 'experiment.json'), 'w+') as f:
        json.dump(exp_params.__dict__, f)
    dataset, masker = load_dataset(exp_params, output_dir=temp_folder)
    loading_parameters = set([(estimator.compression_type,
                               estimator.reduction_ratio)
                              for estimator in estimators])
    full_dict_list = Parallel(n_jobs=exp_params.n_jobs)(
        delayed(single_drop_memmap)(exp_params, temp_folder, index, dataset,
                                    masker, compression_type,
                                    reduction_ratio)
        for index, (compression_type, reduction_ratio) in
        enumerate(loading_parameters))
    data = pd.DataFrame(full_dict_list, columns=['compression_type',
                                                 'reduction_ratio',
                                                 'filename',
                                                 'random_state',
                                                 'math_time',
                                                 'io_time',
                                                 'n_samples',
                                                 'n_voxels'])
    data.sort_index(by=['compression_type',
                        'reduction_ratio'], inplace=True)
    data.to_csv(join(temp_folder, 'data.csv'))
    return temp_folder


def align_single(masker, stack_base, results_dir, exp_int_index, index,
                 sub_df):
    stack_target = np.concatenate(masker.transform(sub_df['components']))
    aligned = _align_one_to_one_flat(stack_base, stack_target)
    filename = join(results_dir, 'aligned_%i.nii.gz' % exp_int_index)
    masker.inverse_transform(aligned).to_filename(filename)
    corr = _spatial_correlation_flat(aligned, stack_base)
    return index, np.mean(corr.diagonal()), filename


def analyse(output_dir, n_jobs=1):
    results_dir = join(output_dir, 'stability')
    if not exists(results_dir):
        os.mkdir(results_dir)
    results = pd.read_csv(join(output_dir, 'results.csv'), index_col=0)

    results['score'] = pd.Series(np.zeros(len(results)), results.index)
    results['aligned_filename'] = pd.Series("", index=results.index)

    mask = check_niimg(join(output_dir, 'mask_img.nii.gz'))
    masker = MultiNiftiMasker(mask_img=mask).fit()
    results.set_index(['estimator_type', 'compression_type', 'reduction_ratio',
                       'alpha', 'random_state'], inplace=True)
    print(
        '[Experiment] Performing Hungarian alg. and computing correlation score')

    stack_base = np.concatenate(
        masker.transform(results.loc[results['reference'], 'components']))
    masker.inverse_transform(stack_base).to_filename(
        join(results_dir, 'base.nii.gz'))
    res_list = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(align_single)(masker, stack_base, results_dir, exp_int_index,
                              index, sub_df)
        for exp_int_index, (index, sub_df) in enumerate(results.groupby(
            level=['estimator_type', 'compression_type', 'reduction_ratio',
                   'alpha'])))
    for index, score, aligned_filename in res_list:
        results.loc[index, 'score'] = score
        results.loc[index, 'aligned_filename'] = aligned_filename

    time_v_corr = results.drop('components', axis='columns')
    time_v_corr.reset_index(level='random_state', drop=True, inplace=True)

    # Selection best scoring alpha for each parameter set
    indices = time_v_corr.groupby(level=['estimator_type',
                                         'compression_type',
                                         'reduction_ratio']).apply(
        lambda x: x['score'].idxmax())
    time_v_corr = time_v_corr.loc[indices]
    time_v_corr.reset_index(level='alpha', drop=False, inplace=True)
    # Mean over random_state
    time_v_corr = time_v_corr.groupby(level=['estimator_type',
                                             'compression_type',
                                             'reduction_ratio']).agg(
        {'math_time': [np.mean, np.std],
         'io_time': [np.mean, np.std],
         'alpha': 'last',
         'reference': 'last',
         'aligned_filename': 'last',
         'score': 'last'})

    time_v_corr.to_csv(join(results_dir, 'time_v_corr.csv'))


def align_incr_single(masker, base_list, this_slice, n_exp, index, sub_df):
    target_list = masker.transform(sub_df['components'][this_slice])
    base = np.concatenate(base_list[:(n_exp + 1)])
    target = np.concatenate(target_list[:(n_exp + 1)])
    aligned = _align_one_to_one_flat(base, target)
    return index, n_exp, np.trace(
        _spatial_correlation_flat(aligned, base)) / len(base)


def analyse_incr(output_dir, n_jobs=1, n_run_var=1):
    results_dir = join(output_dir, 'stability')
    results = pd.read_csv(join(output_dir, 'results.csv'), index_col=0)
    results.reset_index(inplace=True)
    results.set_index(
        ['estimator_type', 'compression_type', 'reduction_ratio', 'alpha'],
        inplace=True)

    time_v_corr = pd.read_csv(join(results_dir, 'time_v_corr.csv'),
                              index_col=range(3), header=[0, 1])
    time_v_corr.reset_index(inplace=True)

    time_v_corr.set_index(
        ['estimator_type', 'compression_type', 'reduction_ratio',
         ('alpha', 'last')], inplace=True)
    time_v_corr.index = time_v_corr.index.set_names('alpha', level=3)

    joined_results = results.join(time_v_corr, how='inner', rsuffix='_mean')
    joined_results.reset_index(inplace=True)
    joined_results.set_index(
        ['estimator_type', 'compression_type', 'reduction_ratio',
         'random_state'], inplace=True)

    mask = check_niimg(join(output_dir, 'mask_img.nii.gz'))
    masker = MultiNiftiMasker(mask_img=mask).fit()
    n_exp = results['reference'].sum()

    slices = gen_even_slices(n_exp, n_run_var)
    incr_stability = []
    for this_slice in slices:
        base_list = masker.transform(
            results.loc[results['reference'], 'components'][this_slice])

        this_incr_stability = pd.DataFrame(columns=np.arange(len(base_list)),
                                           index=joined_results.index)
        res = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(align_incr_single)(masker, base_list, this_slice,
                                       n_exp, index, sub_df)
            for index, sub_df in
            joined_results.groupby(level=['estimator_type',
                                          'compression_type',
                                          'reduction_ratio'])
            for n_exp in range(len(base_list)))
        for index, n_exp, score in res:
            this_incr_stability.loc[index, n_exp] = score
        this_incr_stability = this_incr_stability.groupby(
            level=['estimator_type',
                   'compression_type',
                   'reduction_ratio']).last()
        incr_stability.append(this_incr_stability)

    cat_stability = pd.concat(incr_stability, keys=np.arange(3), names=['run',
                                                                        'estimator_type',
                                                                        'compression_type',
                                                                        'reduction_ratio'],
                              join='inner')

    agg_incr_stability = cat_stability.groupby(level=['estimator_type',
                                                      'compression_type',
                                                      'reduction_ratio']).agg(
        [np.mean, np.std])
    agg_incr_stability.to_csv(join(results_dir, 'agg_incr_stability.csv'))
    time_v_corr.reset_index(level='alpha', inplace=True)
    full = pd.concat([time_v_corr, agg_incr_stability], axis=1)

    full.to_csv(join(results_dir, 'full.csv'))


def plot_incr(output_dir, reduction_ratio=0.2):
    from mpl_utils import plt, figsize
    results_dir = join(output_dir, 'stability')
    figures_dir = join(output_dir, 'figures')
    if not exists(figures_dir):
        os.mkdir(figures_dir)
    time_v_corr = pd.read_csv(join(results_dir, 'full.csv'),
                              index_col=range(3), header=[0, 1])

    n_exp = int(time_v_corr.columns.get_level_values(0)[-1])
    idx = pd.IndexSlice
    time_v_corr = pd.concat([time_v_corr.loc[idx[:, :, reduction_ratio], :],
                             time_v_corr.loc[idx[:, 'subsample', 1], :]],
                            axis=0)
    plt.figure(figsize=figsize(1))
    # Ultra ugly
    for index, sub_df in time_v_corr.groupby(
            level=['estimator_type', 'compression_type', 'reduction_ratio']):
        mean_score = sub_df[[(str(i), 'mean') for i in np.arange(n_exp + 1)]]
        std_score = sub_df[[(str(i), 'std') for i in np.arange(n_exp + 1)]]
        label = '%s %s' % (mean_score.index.get_level_values(1)[0],
                           mean_score.index.get_level_values(2)[0])
        plot_with_error(plt, np.arange(mean_score.shape[1]),
                        mean_score.values[0],
                        yerr=std_score.values[0], label=label)
    plt.legend(loc='lower right')
    plt.xlabel('Number of experiments')
    plt.ylabel('Baseline reproduction')
    plt.savefig(join(figures_dir, 'incr_stability.pdf'))
    plt.savefig(join(figures_dir, 'incr_stability.pgf'))


def plot_with_error(plt, x, y, yerr=0, **kwargs):
    plot = plt.plot(x, y, **kwargs)
    plt.fill_between(x, (y + yerr),
                     (y - yerr), alpha=0.3,
                     color=plot[0].get_color())


def plot_median_maps(output_dir, reduction_ratio=0.1):
    mask = check_niimg(join(output_dir, 'mask_img.nii.gz'))
    masker = MultiNiftiMasker(mask_img=mask).fit()
    results_dir = join(output_dir, 'stability')
    median_dir = join(results_dir, 'median')
    if not os.exists(median_dir):
        os.mkdir(median_dir)
    figures_dir = join(output_dir, 'figures')
    if not exists(figures_dir):
        os.mkdir(figures_dir)
    time_v_corr = pd.read_csv(join(results_dir, 'full.csv'),
                              index_col=range(3), header=[0, 1])
    time_v_corr.rename(columns=convert_litteral_int_to_int, inplace=True)
    n_exp = time_v_corr.columns.get_level_values(0)[-1]
    idx = pd.IndexSlice
    base_components = join(results_dir, 'base.nii.gz')
    target_components = \
    pd.concat([time_v_corr.loc[idx[:, :, reduction_ratio], :],
               time_v_corr.loc[idx[:, 'subsample', 1], :]],
              axis=0)['aligned_filename']
    aligned_target_components = align_many_to_one_nii(masker, base_components,
                                                      target_components)
    aligned_series = pd.Series("", index=target_components.index)
    for i, (index, aligned_components) in enumerate(
            zip(target_components.index,
                aligned_target_components)):
        corr = np.diagonal(
            spatial_correlation(masker, base_components, aligned_components))
        i = np.argsort(corr)[len(corr) / 2]
        median_img = index_img(aligned_components, i)
        median_filename = join(median_dir, 'median_%i.nii.gz' % i)
        median_img.to_filename(median_filename)
        aligned_series.append(median_filename)
    aligned_series.to_csv(join(median_dir, 'csv'))


def plot_full(output_dir):
    from mpl_utils import plt, figsize
    results_dir = join(output_dir, 'stability')
    figures_dir = join(output_dir, 'figures')
    if not exists(figures_dir):
        os.mkdir(figures_dir)
    time_v_corr = pd.read_csv(join(results_dir, 'full.csv'),
                              index_col=range(3), header=[0, 1])
    time_v_corr.rename(columns=convert_litteral_int_to_int, inplace=True)
    n_exp = time_v_corr.columns.get_level_values(0)[-1]

    ref_time = \
        time_v_corr.loc[
            time_v_corr[('reference', 'last')], ('math_time', 'mean')][
            0]
    ref_reproduction = time_v_corr.loc[
        ('DictLearning', 'subsample', 1.), (n_exp, 'mean')]
    ref_std = time_v_corr.loc[
        ('DictLearning', 'subsample', 1.), (n_exp, 'std')]
    # ref_reproduction = time_v_corr.loc[
    #     ('DictLearning', 'subsample', 1.), ('score', 'last')]
    fig = []
    for i in range(3):
        fig.append(plt.figure(figsize=figsize(1)))
    for index, sub_df in time_v_corr[
                time_v_corr[('reference', 'last')] == False].groupby(
        level=['estimator_type',
               'compression_type']):
        plt.figure(fig[0].number, axis='square')
        # plt.plot(np.linspace(0, 1, 10), np.linspace(0, ref_reproduction, 10),
        #          '--', color='black',
        #          label='Time / accuracy tradeoff')
        plt.errorbar(sub_df[('math_time', 'mean')] / ref_time,
                     sub_df[(n_exp, 'mean')],
                     yerr=sub_df[(n_exp, 'std')],
                     # label=sub_df.index.get_level_values(1)[0],
                     xerr=sub_df[('math_time', 'std')] / ref_time,
                     marker='o')
        plt.xlim([0.1, 1])
        plt.ylim([0., 0.5])

        plt.figure(fig[1].number)

        plot_with_error(plt, sub_df.index.get_level_values(2),
                        sub_df[(n_exp, 'mean')],
                        yerr=sub_df[(n_exp, 'std')],
                        label=sub_df.index.get_level_values(1)[0], marker='o')
        plt.figure(fig[2].number)

        plot_with_error(plt, sub_df.index.get_level_values(2),
                        sub_df[('math_time', 'mean')],
                        yerr=sub_df[('math_time', 'std')],
                        label=sub_df.index.get_level_values(1)[0], marker='o')
    plt.figure(fig[0].number)
    plot_with_error(plt, np.linspace(0, 1, 10),
                    ref_reproduction * np.ones(10),
                    yerr=ref_std * np.ones(10),
                    color='red')
    plt.legend(['Range finder', 'Subsample', 'No reduction'], loc='lower left')
    plt.ylabel('Mean overlap')
    plt.xlabel('Time (relative to baseline)')
    plt.savefig(join(figures_dir, 'time_v_corr.pdf'))
    plt.savefig(join(figures_dir, 'time_v_corr.pgf'))

    plt.figure(fig[1].number)
    plt.legend(loc='lower right')
    plt.ylabel('Baseline reproduction')
    plt.xlabel('Reduction ratio')
    plt.savefig(join(figures_dir, 'corr.pdf'))
    plt.savefig(join(figures_dir, 'corr.pgf'))

    plt.figure(fig[2].number)
    plt.legend(loc='lower right')
    plt.ylabel('Time')
    plt.xlabel('Reduction ratio')
    plt.savefig(join(figures_dir, 'time.pdf'))
    plt.savefig(join(figures_dir, 'time.pgf'))


def convert_litteral_int_to_int(x):
    try:
        return int(x)
    except ValueError:
        return x


def convert_nii_to_pdf(output_dir, n_jobs=1):
    import matplotlib as mpl
    mpl.use('PDF')
    from nilearn_sandbox.plotting.pdf_plotting import plot_to_pdf
    list_nii = glob.glob(join(output_dir, 'stability', "*.nii.gz"))
    print(list_nii)
    list_pdf = []
    for this_nii in list_nii:
        this_pdf = this_nii[:-7] + ".pdf"
        list_pdf.append(this_pdf)
    print(list_pdf)
    Parallel(n_jobs=1)(delayed(plot_to_pdf)(this_nii, this_pdf)
                            for this_nii, this_pdf in zip(list_nii, list_pdf))


def clean_memory():
    try:
        shutil.rmtree(expanduser('~/nilearn_cache/joblib/sklearn'))
    except:
        pass

    try:
        shutil.rmtree(expanduser('~/nilearn_cache/joblib/scipy'))
    except:
        pass
