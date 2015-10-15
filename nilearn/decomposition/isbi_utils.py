import fnmatch

import matplotlib.pyplot as plt

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
from pandas import IndexSlice as idx

from joblib import Parallel, delayed, Memory

from sklearn.utils import gen_even_slices
from sklearn.base import clone

from nilearn_sandbox._utils.map_alignment import _align_one_to_one_flat, \
    _spatial_correlation_flat, spatial_correlation, align_many_to_one_nii

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


def run_single(index, slice_index, estimator, dataset, output_dir, reference,
               this_slice,
               data=None):
    exp_output = join(output_dir, "experiment_%i_%i" % (index, slice_index))
    os.mkdir(exp_output)
    if type(estimator).__name__:
        debug_folder = join(exp_output, 'debug')
        os.mkdir(debug_folder)
    estimator.set_params(debug_folder=debug_folder)
    print('[Example] Learning maps')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if data is not None:
            memmap_data = np.memmap(data['components'], dtype='float64',
                                    order='F', mode='r',
                                    shape=(
                                        data['n_samples'], data['n_voxels']))
            # For out-of-core computation
            estimator.set_params(in_memory=False)
            subject_limits = np.load(data['subject_limits'])
            estimator._raw_fit(memmap_data[subject_limits[this_slice.start]:
                                           subject_limits[this_slice.stop]])
            del memmap_data
        else:
            estimator.fit(dataset[this_slice])
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
                       'slice': str(this_slice),
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
    slices = gen_even_slices(len(dataset), exp_params.n_slices)
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
        delayed(run_single)(index, slice_index, estimator, dataset, output_dir,
                            reference, this_slice,
                            data=data)
        for slice_index, this_slice in enumerate(slices)
        for index, ((estimator, reference), data) in
        enumerate(zip(exp_estimators, estimators_data)))
    results = pd.DataFrame(full_dict_list, columns=['estimator_type',
                                                    'compression_type',
                                                    'reduction_ratio',
                                                    'alpha',
                                                    'slice',
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
                           'slice',
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
                                                    'slice',
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
                           'slice',
                           'random_state'], inplace=True)
    results.to_csv(join(output_dir, 'results.csv'))


def drop_memmap_single(exp_params, temp_folder, index, dataset,
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
    subject_limits_filename = join(temp_folder, filename + '_limits.npy')
    np.save(subject_limits_filename,
            mask_reducer.subject_limits_)
    single_run_dict = {'compression_type': compression_type,
                       'reduction_ratio':
                           reduction_ratio,
                       'components': join(temp_folder, filename),
                       'subject_limits': subject_limits_filename,
                       'random_state': 0,
                       'math_time': mask_reducer.time_[0],
                       'io_time': mask_reducer.time_[1],
                       'n_samples': mask_reducer.data_.shape[0],
                       'n_voxels': mask_reducer.data_.shape[1]
                       }

    with open(join(temp_folder, filename + '.json'), 'w+') as f:
        json.dump(single_run_dict, f)
    return single_run_dict


def drop_memmmap(estimators, exp_params):
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
        delayed(drop_memmap_single)(exp_params, temp_folder, index, dataset,
                                    masker, compression_type,
                                    reduction_ratio)
        for index, (compression_type, reduction_ratio) in
        enumerate(loading_parameters))
    data = pd.DataFrame(full_dict_list, columns=['compression_type',
                                                 'reduction_ratio',
                                                 'components',
                                                 'subject_limits',
                                                 'random_state',
                                                 'math_time',
                                                 'io_time',
                                                 'n_samples',
                                                 'n_voxels'])
    data.sort_index(by=['compression_type',
                        'reduction_ratio'], inplace=True)
    data.to_csv(join(temp_folder, 'data.csv'))
    return temp_folder


def analyse_single(masker, stack_base, results_dir, num, index,
                   random_state_df, limit, cache_dir):
    stack_target = np.concatenate(
        masker.transform(random_state_df['components'][:limit]))
    aligned = _align_one_to_one_flat(stack_base, stack_target, mem=Memory(cache_dir=cache_dir))
    filename = join(results_dir, 'aligned_%i.nii.gz' % num)
    masker.inverse_transform(aligned).to_filename(filename)
    corr = _spatial_correlation_flat(aligned, stack_base)
    return index, np.mean(corr.diagonal()), filename


def analyse(exp_params, output_dir, n_jobs=1, limit=10):
    results_dir = join(output_dir, 'stability')
    cache_dir = exp_params.cache_dir
    if not exists(results_dir):
        os.mkdir(results_dir)
    results = pd.read_csv(join(output_dir, 'results.csv'), index_col=0)
    results.set_index(['reference', 'estimator_type', 'compression_type',
                       'reduction_ratio',
                       'alpha', 'random_state'], inplace=True)
    results.sortlevel(inplace=True)
    results['score'] = pd.Series(np.zeros(len(results)), results.index)
    results['aligned'] = pd.Series("", index=results.index)

    mask = check_niimg(join(output_dir, 'mask_img.nii.gz'))
    masker = MultiNiftiMasker(mask_img=mask).fit()

    print('[Experiment] Performing Hungarian alg.'
          ' and computing correlation score')

    stack_base = np.concatenate(
        masker.transform(results.loc[True]['components'][:limit]))
    masker.inverse_transform(stack_base).to_filename(
        join(results_dir, 'base.nii.gz'))

    res = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(analyse_single)(masker, stack_base, results_dir, num,
                                index, random_state_df, limit, cache_dir)
        for num, (index, random_state_df) in enumerate(results.groupby(
            level=['reference', 'estimator_type', 'compression_type',
                   'reduction_ratio',
                   'alpha'])))
    for index, score, aligned_filename in res:
        results.loc[index, 'score'] = score
        results.loc[index, 'aligned'] = aligned_filename

    scores = results.drop('components', axis='columns')
    scores.reset_index(level='random_state', drop=True, inplace=True)

    # Selection best scoring alpha for each parameter set
    indices = scores.groupby(level=['reference',
                                    'estimator_type',
                                    'compression_type',
                                    'reduction_ratio']).apply(
        lambda x: x['score'].idxmax())

    scores = scores.loc[indices.values]
    scores.reset_index(level='alpha', drop=False, inplace=True)
    # Mean over random_state
    scores = scores.groupby(level=['reference',
                                   'estimator_type',
                                   'compression_type',
                                   'reduction_ratio']).agg(
        {'math_time': [np.mean, np.std],
         'io_time': [np.mean, np.std],
         'load_math_time': 'last',
         'load_io_time': 'last',
         'alpha': 'last',
         'aligned': 'last',
         'score': 'last'})

    scores.to_csv(join(results_dir, 'scores.csv'))


def align_num_exp_single(masker, base_list, this_slice, n_exp, index,
                         random_state_df):
    target_list = masker.transform(random_state_df['components'][this_slice])
    base = np.concatenate(base_list[:(n_exp + 1)])
    target = np.concatenate(target_list[:(n_exp + 1)])
    aligned = _align_one_to_one_flat(base, target)
    return index, n_exp, np.trace(
        _spatial_correlation_flat(aligned, base)) / len(base)


def analyse_num_exp(output_dir, n_jobs=1, n_run_var=1):
    results_dir = join(output_dir, 'stability')
    results = pd.read_csv(join(output_dir, 'results.csv'), index_col=0)
    results.set_index(['reference', 'estimator_type', 'compression_type',
                       'reduction_ratio',
                       'alpha'], inplace=True)
    results.sortlevel(inplace=True)

    scores = pd.read_csv(join(results_dir, 'scores.csv'),
                         index_col=range(4), header=[0, 1])
    scores.reset_index(inplace=True)
    scores.set_index(
        ['reference', 'estimator_type', 'compression_type',
         'reduction_ratio',
         ('alpha', 'last')], inplace=True)
    scores.index = scores.index.set_names('alpha', level=4)
    scores.sortlevel(inplace=True)

    results_score = results.join(scores, how='inner', rsuffix='_mean')
    results_score.reset_index('alpha', drop=False, inplace=True)
    # alpha is now a column of joinded results
    results_score.sortlevel(0)

    mask = check_niimg(join(output_dir, 'mask_img.nii.gz'))
    masker = MultiNiftiMasker(mask_img=mask).fit()
    n_exp = results.loc[True]['random_state'].count() / n_run_var

    slices = gen_even_slices(n_exp, n_run_var)
    n_exp /= n_run_var
    # Number of experiment = number of reference experiment

    score_num_exp = []

    for this_slice in slices:
        base_list = masker.transform(
            results.loc[True]['components'][this_slice])

        this_stability = pd.DataFrame(columns=np.arange(len(base_list)),
                                      index=results_score.index)
        res = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(align_num_exp_single)(masker, base_list, this_slice,
                                          i, index, random_state_df)
            for index, random_state_df in
            results_score.groupby(level=['reference',
                                         'estimator_type',
                                         'compression_type',
                                         'reduction_ratio']))
        for index, n_exp, score in res:
            this_stability.loc[index, n_exp] = score

        this_stability = this_stability.groupby(
            level=['reference',
                   'estimator_type',
                   'compression_type',
                   'reduction_ratio']).last()
        score_num_exp.append(this_stability)

    score_num_exp = pd.concat(score_num_exp, keys=np.arange(3),
                              names=['run', 'reference', 'estimator_type',
                                     'compression_type',
                                     'reduction_ratio'],
                              join='inner')

    score_num_exp = score_num_exp.groupby(level=['reference',
                                                 'estimator_type',
                                                 'compression_type',
                                                 'reduction_ratio']).agg(
        [np.mean, np.std])
    score_num_exp.to_csv(join(results_dir, 'scores_num_exp.csv'))
    scores.reset_index(level='alpha', inplace=True)
    scores_extended = pd.concat([scores, score_num_exp], axis=1)

    scores_extended.to_csv(join(results_dir, 'scores_extended.csv'))


def analyse_median_maps(output_dir, reduction_ratio=0.1):
    results_dir = join(output_dir, 'stability')

    median_dir = join(results_dir, 'median')

    if not exists(median_dir):
        os.mkdir(median_dir)

    scores_extended = pd.read_csv(join(results_dir, 'scores.csv'),
                                  index_col=range(4), header=[0, 1])
    scores_extended.rename(columns=convert_litteral_int_to_int, inplace=True)
    n_exp = scores_extended.columns.get_level_values(0)[-1]

    mask = check_niimg(join(output_dir, 'mask_img.nii.gz'))
    masker = MultiNiftiMasker(mask_img=mask).fit()

    base_components = scores_extended.loc[True]['aligned', 'last'][0]
    target_components = pd.concat((scores_extended.loc[idx[False, :,
                                                       ['subsample',
                                                        'range_finder'],
                                                       reduction_ratio], :],
                                   scores_extended.loc[
                                   idx[False, :, 'subsample', 1.],
                                   :]))[('aligned', 'last')]

    aligned_target_components = align_many_to_one_nii(masker, base_components,
                                                      target_components)

    median_series = pd.Series("", index=target_components.index)
    corr = np.diagonal(
        spatial_correlation(masker, base_components,
                            aligned_target_components[-1]))
    i = np.argsort(corr)[len(corr) / 2]
    median_img = index_img(base_components, i)
    median_filename = join(median_dir, 'base.nii.gz')
    median_img.to_filename(median_filename)
    for j, (index, aligned_components) in enumerate(
            zip(target_components.index,
                aligned_target_components)):
        median_img = index_img(aligned_components, i)
        median_filename = join(median_dir, 'median_%i.nii.gz' % j)
        median_img.to_filename(median_filename)
        median_series.loc[index] = median_filename
    target_components['median'] = median_series
    median_series.to_csv(join(median_dir, 'median.csv'))


def plot_num_exp(output_dir, reduction_ratio=0.1):
    results_dir = join(output_dir, 'stability')
    figures_dir = join(output_dir, 'figures')
    if not exists(figures_dir):
        os.mkdir(figures_dir)

    scores_extended = pd.read_csv(join(results_dir, 'scores_extended.csv'),
                                  index_col=range(4), header=[0, 1])
    scores_extended.rename(columns=convert_litteral_int_to_int, inplace=True)
    n_exp = scores_extended.columns.get_level_values(0)[-1]

    scores_extended = pd.concat((scores_extended.loc[idx[False, :,
                                                     ['subsample',
                                                      'range_finder'],
                                                     reduction_ratio], :],
                                 scores_extended.loc[
                                 idx[False, :, 'subsample', 1.],
                                 :]))
    # scores_extended.fillna(0, inplace=True)
    labels = ['Range finder', 'Subsampling', 'Baseline']
    plt.figure()
    for j, (index, exp_df) in enumerate(scores_extended.iterrows()):
        mean_score = exp_df[[(i, 'mean') for i in np.arange(n_exp + 1)]].values.astype('float')
        std_score = exp_df[[(i, 'std') for i in np.arange(n_exp + 1)]].values.astype('float')
        plot_with_error(np.arange(len(mean_score)),
                        mean_score,
                        yerr=std_score, label=labels[j], marker='o')
    plt.legend(loc='lower right', ncol=2)
    plt.xlabel('Number of experiments')
    plt.ylabel('Baseline reproduction')
    plt.savefig(join(figures_dir, 'incr_stability.pgf'))
    plt.savefig(join(figures_dir, 'incr_stability.pdf'))


def plot_with_error(x, y, yerr=0, **kwargs):
    plot = plt.plot(x, y, **kwargs)
    plt.fill_between(x, (y + yerr),
                     (y - yerr), alpha=0.3,
                     color=plot[0].get_color())


def plot_full(output_dir):
    results_dir = join(output_dir, 'stability')
    figures_dir = join(output_dir, 'figures')
    if not exists(figures_dir):
        os.mkdir(figures_dir)

    scores_extended = pd.read_csv(join(results_dir, 'scores_extended.csv'),
                                  index_col=range(4), header=[0, 1])
    scores_extended.rename(columns=convert_litteral_int_to_int, inplace=True)
    n_exp = scores_extended.columns.get_level_values(0)[-1]

    ref_time = scores_extended.loc[
        idx[False, 'DictLearning', 'subsample', 1], ('math_time', 'mean')]
    ref_time += scores_extended.loc[
        idx[False, 'DictLearning', 'subsample', 1], ('load_math_time', 'last')]

    ref_reproduction = scores_extended.loc[
        idx[False, 'DictLearning', 'subsample', 1], (n_exp, 'mean')]
    ref_std = scores_extended.loc[
        idx[False, 'DictLearning', 'subsample', 1], (n_exp, 'std')]

    scores_extended = scores_extended.loc[idx[False, :,
                                          ['subsample',
                                           'range_finder'],
                                          ], :]
    fig = []
    for i in range(3):
        fig.append(plt.figure())
    name_index = {'range_finder': 'Range finder',
                  'subsample': 'Subsample'}
    plt.figure(fig[0].number)
    plot_with_error(np.linspace(0, 1, 10),
                    ref_reproduction * np.ones(10),
                    yerr=ref_std,
                    label='Baseline')
    for index, exp_df in scores_extended.groupby(level=['estimator_type',
                                                        'compression_type']):
        plt.figure(fig[0].number)
        score = exp_df[n_exp]
        total_time = pd.DataFrame(exp_df['math_time'])
        total_time.loc[:, 'mean'] += exp_df['load_math_time', 'last']
        total_time /= ref_time
        reduction_ratio = exp_df.index.get_level_values(3).values
        compression_type = index[1]
        label = name_index[compression_type]

        plt.errorbar(total_time['mean'],
                     score['mean'].values,
                     xerr=total_time['std'],
                     yerr=score['std'].values,
                     label=label,
                     marker='o')
        plt.xlim([0.1, 1])
        plt.ylim([0., 1])

        plt.figure(fig[1].number)

        plot_with_error(reduction_ratio,
                        score['mean'].values,
                        yerr=score['std'].values,
                        label=label, marker='o')
        plt.figure(fig[2].number)

        plot_with_error(reduction_ratio,
                        score['mean'].values,
                        yerr=score['std'].values,
                        label=label, marker='o')
    plt.figure(fig[0].number)

    plt.legend(loc='lower left')
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
    from nilearn_sandbox.plotting.pdf_plotting import plot_to_pdf
    list_nii = glob.glob(join(output_dir, "*.nii.gz"))
    print(list_nii)
    list_pdf = []
    for this_nii in list_nii:
        this_pdf = this_nii[:-7] + ".pdf"
        list_pdf.append(this_pdf)
    print(list_pdf)
    Parallel(n_jobs=n_jobs)(delayed(plot_to_pdf)(this_nii, this_pdf)
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
