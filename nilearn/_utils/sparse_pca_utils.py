import collections
import datetime
import fnmatch
import json
import os
import warnings
from os.path import join, exists

import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed, Memory
from nilearn_sandbox import datasets as datasets_sandbox
from nilearn_sandbox._utils.map_alignment import _align_one_to_one_flat, \
    spatial_correlation, align_many_to_one_nii
from nilearn_sandbox._utils.map_alignment import _spatial_correlation_flat
from sklearn import clone
from sklearn.utils import gen_even_slices

from nilearn import datasets
from nilearn._utils import check_niimg
from nilearn.decomposition.base import BaseDecomposition
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_prob_atlas, plot_stat_map
from pandas import IndexSlice as idx

import matplotlib.pyplot as plt


Experiment = collections.namedtuple('Experiment',
                                    ['dataset_name',
                                     'n_subjects',
                                     'smoothing_fwhm',
                                     'dict_init',
                                     'output_dir',
                                     'cachedir',
                                     'data_dir',
                                     'n_slices',
                                     'n_jobs',
                                     'parallel_exp',
                                     # 'n_epochs',
                                     'n_runs'])


def load_dataset(exp_params, output_dir=None):
    n_subjects = exp_params.n_subjects
    data_dir = exp_params.data_dir

    if exp_params.dataset_name == 'adhd':
        dataset = datasets.fetch_adhd(n_subjects=
                                      min(40, n_subjects)).func
        mask = join(exp_params.data_dir, 'ADHD_mask', 'mask_img.nii.gz')
    elif exp_params.dataset_name == 'hcp':
        dataset = datasets_sandbox.fetch_hcp_rest(n_subjects=n_subjects,
                                                  data_dir=data_dir).func
        mask = join(exp_params.data_dir, 'HCP_mask', 'mask_img.nii.gz')
    elif exp_params.dataset_name == 'adni':
        dataset = datasets_sandbox.fetch_adni_longitudinal_rs_fmri_DARTEL().func
        dataset = dataset[:exp_params.n_subjects]
        mask = datasets_sandbox.fetch_adni_masks().fmri
    if output_dir is not None:
        check_niimg(mask).to_filename(join(output_dir, 'mask_img.nii.gz'))

    print('[Experiment] Computing global mask')
    smoothing_fwhm = exp_params.smoothing_fwhm
    cachedir = exp_params.cachedir
    n_jobs = exp_params.n_jobs

    decomposition_estimator = BaseDecomposition(smoothing_fwhm=
                                                smoothing_fwhm,
                                                memory=cachedir,
                                                mask=mask,
                                                memory_level=2,
                                                verbose=10,
                                                n_jobs=n_jobs)
    decomposition_estimator.fit(dataset)

    masker = decomposition_estimator.masker_
    masker.mask_img_.get_data()

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
    elif isinstance(exp_params.dict_init, int):
        dict_init = None
        n_components = exp_params.dict_init
    else:
        raise ValueError('Init not supported')
    return dict_init, n_components


def yield_estimators(estimators, exp_params, masker, dict_init, n_components):
    smoothing_fwhm = exp_params.smoothing_fwhm
    # n_epochs = exp_params.n_epochs
    n_runs = exp_params.n_runs
    cachedir = exp_params.cachedir
    n_jobs = 1 if exp_params.parallel_exp else exp_params.n_jobs
    for random_state in range(n_runs):
        for i, estimator in enumerate(estimators):
            if i == 0:
                reference = True
                rs_offset = 1000
            else:
                reference = False
                rs_offset = 0
            estimator = clone(estimator)
            estimator.set_params(mask=masker,
                                 smoothing_fwhm=smoothing_fwhm,
                                 # n_epochs=n_epochs,
                                 n_jobs=n_jobs,
                                 dict_init=dict_init,
                                 n_components=n_components,
                                 memory_level=2, memory=cachedir,
                                 verbose=3,
                                 random_state=random_state + rs_offset)
            yield estimator, reference


def run_single(index, slice_index, estimator, dataset, output_dir,
               this_slice,
               reference,
               data=None):
    exp_output = join(output_dir, "experiment_%i_%i" % (index, slice_index))
    os.mkdir(exp_output)
    if type(estimator).__name__:
        debug_folder = join(exp_output, 'debug')
        os.mkdir(debug_folder)
    estimator.set_params(debug_folder=debug_folder)
    single_run_dict = {'reference': reference,
                       'estimator_type': type(estimator).__name__,
                       'reduction_method': estimator.reduction_method,
                       'reduction_ratio':
                           estimator.reduction_ratio,
                       'feature_ratio': estimator.feature_ratio,
                       'alpha': estimator.alpha,
                       # 'support': estimator.support,
                       'random_state': estimator.random_state,
                       'slice': str(this_slice),
                       }
    with open(join(exp_output, 'results.json'), 'w+') as f:
        json.dump(single_run_dict, f)
    print('[Example] Learning maps')
    cut = int(this_slice.stop * 9 / 10)
    train = slice(0, cut)
    test = slice(cut, this_slice.stop)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        estimator.fit(dataset[this_slice][train],
                      probe=dataset[this_slice][test])
    print('[Example] Dumping results')
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_filename = join(exp_output, 'components.nii.gz')
    components_img.to_filename(components_filename)
    math_time = estimator.time_[0]
    io_time = estimator.time_[1]
    single_run_dict = {'reference': reference,
                       'estimator_type': type(estimator).__name__,
                       'reduction_method': estimator.reduction_method,
                       'reduction_ratio':
                           estimator.reduction_ratio,
                       'feature_ratio': estimator.feature_ratio,
                       'alpha': estimator.alpha,
                       'random_state': estimator.random_state,
                       'slice': str(this_slice),
                       # Columns
                       'components': components_filename,
                       'math_time': math_time,
                       'io_time': io_time,
                       }
    with open(join(exp_output, 'results.json'), 'w+') as f:
        json.dump(single_run_dict, f)
    return single_run_dict


def run(estimators, exp_params, temp_folder=None):
    output_dir = join(exp_params.output_dir,
                      datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                       '-%M-%S'))
    os.makedirs(output_dir)
    with open(join(output_dir, 'experiment.json'), 'w+') as f:
        json.dump(exp_params.__dict__, f)

    dataset, masker = load_dataset(exp_params, output_dir=output_dir)

    dataset_series = pd.Series(dataset)
    dataset_series.to_csv(join(output_dir, 'dataset.csv'))

    dict_init, n_components = check_init(exp_params)

    if dict_init is not None:
        check_niimg(dict_init).to_filename(join(output_dir,
                                                'dict_init.nii.gz'))
    res = list(yield_estimators(estimators,
                                           exp_params,
                                           masker,
                                           dict_init,
                                           n_components))
    exp_estimators, references = zip(*res)
    slices = list(gen_even_slices(len(dataset), exp_params.n_slices))

    if temp_folder is not None:
        estimators_data = pd.read_csv(join(temp_folder, 'data.csv'),
                                      index_col=0)
        estimators_data.reset_index(inplace=True)
        estimators_data.set_index(['reduction_method', 'reduction_ratio'],
                                  inplace=True)
        estimators_data = [estimators_data.loc[estimator.reduction_method,
                                               round(estimator.reduction_ratio,
                                                     3)]
                           for estimator, _ in exp_estimators]
    else:
        estimators_data = [None] * len(exp_estimators)
    if not exp_params.parallel_exp:
        for slice_index, this_slice in enumerate(slices):
            for index, (estimator, data, reference) in enumerate(
                    zip(exp_estimators, estimators_data, references)):
                run_single(index, slice_index, estimator, dataset, output_dir,
                           this_slice,
                           reference,
                           data=data)
    else:
        print('n_jobs : %i' % exp_params.n_jobs)
        Parallel(n_jobs=exp_params.n_jobs, verbose=10)(delayed(
            run_single)(index, slice_index, estimator, dataset, output_dir,
                        this_slice,
                        reference,
                        data=data) for slice_index, this_slice in
                                                       enumerate(slices)
                                                       for
                                                       index, (estimator, data,
                                                               reference)
                                                       in enumerate(
            zip(exp_estimators, estimators_data, references)))
    return output_dir


def gather_results(output_dir):
    full_dict_list = []
    for dirpath, dirname, filenames in os.walk(output_dir):
        for filename in fnmatch.filter(filenames, 'results.json'):
            with open(join(dirpath, filename), 'r') as f:
                exp_dict = json.load(f)
                if exp_dict['reduction_method'] is None:
                    exp_dict['reduction_method'] = 'none'
                full_dict_list.append(exp_dict)
    results = pd.DataFrame(full_dict_list, columns=['reference',
                                                    'estimator_type',
                                                    'reduction_method',
                                                    'reduction_ratio',
                                                    'feature_ratio',
                                                    # 'support',
                                                    'alpha',
                                                    'slice',
                                                    'random_state',
                                                    'math_time', 'io_time',
                                                    'components'])

    results.sort_values(by=['estimator_type',
                            'reduction_method',
                            'reduction_ratio',
                            'feature_ratio',
                            # 'support',
                            'alpha',
                            'slice',
                            'random_state'], inplace=True)
    results.to_csv(join(output_dir, 'results.csv'))


def display_single(output_dir):
    import matplotlib.pyplot as plt
    from nilearn_sandbox.plotting.papaya import papaya_viewer
    plot_prob_atlas(join(output_dir, 'components.nii.gz'),
                    output_file=join(output_dir, 'components.png'))
    papaya_viewer(join(output_dir, 'components.nii.gz'),
                  output_file=join(output_dir,
                                   'components.html'))
    residuals = np.load(join(output_dir, 'debug', 'residuals.npy'))
    plt.plot(np.arange(len(residuals)), residuals)
    plt.savefig(join(output_dir, 'debug', 'residuals.png'))
    for dirpath, dirname, filenames in os.walk(join(output_dir,
                                                    'debug', 'intermediary')):
        for filename in fnmatch.filter(filenames, 'at_*.nii.gz'):
            input_file = join(output_dir, 'debug', 'intermediary',
                              filename)
            output_file = input_file[:-7] + '.png'
            plot_prob_atlas(input_file, output_file=output_file)


def display_all(output_dir):
    exp_dirs = os.listdir(output_dir)
    for exp_dir in fnmatch.filter(exp_dirs, 'experiment_*_*'):
        display_single(join(output_dir, exp_dir))


def analyse_single(masker, stack_base, results_dir, num, index,
                   random_state_df, cachedir):
    stack_target = np.concatenate(
        masker.transform(random_state_df['components']))
    aligned = _align_one_to_one_flat(stack_base, stack_target,
                                     mem=Memory(cachedir=cachedir))
    filename = join(results_dir, 'aligned_%i.nii.gz' % num)
    masker.inverse_transform(aligned).to_filename(filename)
    corr = _spatial_correlation_flat(aligned, stack_base)
    return index, np.trace(corr) / len(corr), filename


def analyse(exp_params, output_dir, n_jobs=1):
    results_dir = join(output_dir, 'stability')
    cachedir = exp_params.cachedir
    if not exists(results_dir):
        os.mkdir(results_dir)
    results = pd.read_csv(join(output_dir, 'results.csv'), index_col=0)
    results.set_index(['reference', 'estimator_type', 'reduction_method',
                       'reduction_ratio',
                       'feature_ratio',
                       'alpha', 'random_state'], inplace=True)
    results.sortlevel(inplace=True)
    results['score'] = pd.Series(np.zeros(len(results)), results.index)
    results['aligned'] = pd.Series("", index=results.index)
    mask = check_niimg(join(output_dir, 'mask_img.nii.gz'))
    masker = MultiNiftiMasker(mask_img=mask).fit()

    print('[Experiment] Performing Hungarian alg.'
          ' and computing correlation score')

    stack_base = np.concatenate(
        masker.transform(results.loc[True]['components']))
    masker.inverse_transform(stack_base).to_filename(
        join(results_dir, 'base.nii.gz'))

    res = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(analyse_single)(masker, stack_base, results_dir, num,
                                index, random_state_df, cachedir)
        for num, (index, random_state_df) in enumerate(results.groupby(
            level=['reference', 'estimator_type', 'reduction_method',
                   'reduction_ratio',
                   'feature_ratio',
                   'alpha'])))
    for index, score, aligned_filename in res:
        results.loc[index, 'score'] = score
        results.loc[index, 'aligned'] = aligned_filename

    scores = results.drop('components', axis='columns')
    scores.reset_index(level='random_state', drop=True, inplace=True)

    # Selection best scoring alpha for each parameter set
    indices = scores.groupby(
        level=['reference', 'estimator_type', 'reduction_method',
               'reduction_ratio',
               'feature_ratio', ]).apply(
        lambda x: x['score'].idxmax())

    scores = scores.loc[indices.values]
    scores.reset_index(level='alpha', drop=False, inplace=True)
    # Mean over random_state
    scores = scores.groupby(
        level=['reference', 'estimator_type', 'reduction_method',
               'reduction_ratio',
               'feature_ratio']).agg(
        {'math_time': [np.mean, np.std],
         'io_time': [np.mean, np.std],
         'alpha': 'last',
         'aligned': 'last',
         'score': 'last'})

    scores.to_csv(join(results_dir, 'scores.csv'))


def align_num_exp_single(masker, base_list, this_slice, n_exp, index,
                         random_state_df, cachedir=None):
    target_list = masker.transform(random_state_df['components'][this_slice])
    base = np.concatenate(base_list[:(n_exp + 1)])
    target = np.concatenate(target_list[:(n_exp + 1)])
    aligned = _align_one_to_one_flat(base, target,
                                     mem=Memory(cachedir=cachedir))
    corr = _spatial_correlation_flat(aligned, base)
    # non_zero = np.sum(np.logical_or(np.any(base, axis=1),
    #                                 np.any(aligned, axis=1)))
    return index, n_exp, (np.trace(corr)) / len(corr)


def analyse_num_exp(exp_params, output_dir, n_jobs=1, n_run_var=1, limit=1000):
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
    # Number of experiment = number of reference experiment
    total_n_exp = min(limit, results.loc[True]['random_state'].count())

    slices = gen_even_slices(total_n_exp, n_run_var)
    total_n_exp = total_n_exp / n_run_var
    score_num_exp = []

    for this_slice in slices:
        base_list = masker.transform(
            results.loc[True]['components'][this_slice])

        this_stability = pd.DataFrame(columns=np.arange(len(base_list)),
                                      index=results_score.index)
        res = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(align_num_exp_single)(masker, base_list, this_slice,
                                          i, index, random_state_df,
                                          cachedir=exp_params.cachedir)
            for index, random_state_df in
            results_score.groupby(level=['reference',
                                         'estimator_type',
                                         'compression_type',
                                         'reduction_ratio'])
            for i in range(0, total_n_exp))
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


def analyse_median_maps(output_dir):
    results_dir = join(output_dir, 'stability')

    median_dir = join(results_dir, 'median')

    if not exists(median_dir):
        os.mkdir(median_dir)

    scores_extended = pd.read_csv(join(results_dir, 'scores.csv'),
                                  index_col=list(range(5)), header=[0, 1])

    mask = check_niimg(join(output_dir, 'mask_img.nii.gz'))
    masker = MultiNiftiMasker(mask_img=mask).fit()

    base_components = scores_extended.loc[True]['aligned', 'last'][0]
    target_components = scores_extended.loc[idx[False, :, :, [1, 2]],
                                            ( 'aligned', 'last')]

    aligned_target_components = align_many_to_one_nii(masker, base_components,
                                                      target_components)

    median_series = pd.Series("", index=target_components.index)
    corr = np.diagonal(
        spatial_correlation(masker, base_components,
                            aligned_target_components[-1]))
    len_non_zero = np.sum(corr != 0.)
    print(len_non_zero)
    i = np.argsort(corr)[::-1][len_non_zero / 2 + 1]
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


def plot_median(output_dir):
    results_dir = join(output_dir, 'stability')
    median_dir = join(results_dir, 'median')
    figures_dir = join(output_dir, 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    median_series = pd.read_csv(join(median_dir, 'median.csv'),
                                index_col=list(range(5)),
                                header=None)
    fig, axes = plt.subplots(2, 4, figsize=(3.38676401384, 1.6), gridspec_kw=dict(hspace=0.3))
    axes = axes.reshape(-1)
    for i, (index, img) in enumerate(median_series.iterrows()):
        plot_stat_map(img.values[0], display_mode='x',
                      cut_coords=[-18],
                      figure=fig,
                      axes=axes[2 * i + 2], colorbar=False,
                      annotate=False)
        plot_stat_map(img.values[0], display_mode='y',
                      cut_coords=[-88],
                      figure=fig,
                      axes=axes[3 + 2 * i], colorbar=False,
                      annotate=False)
        if index[4]== 1:
            label = 'Second run'
        else:
            label = 'Red. ratio: %i' % index[4]
        axes[2 * i + 2].annotate(label, xy=(1.15, 0.), xytext=(0, -5),
                                 xycoords="axes fraction",
                                 textcoords='offset points',
                                 va='center', ha="center")
    plot_stat_map(join(median_dir, 'base.nii.gz'), display_mode='x',
                  cut_coords=1,
                  axes=axes[0], colorbar=False, annotate=False)
    plot_stat_map(join(median_dir, 'base.nii.gz'), display_mode='y',
                  cut_coords=1,
                  axes=axes[1], colorbar=False, annotate=False)

    axes[0].annotate("Reference run", xy=(1.15, 0.), xytext=(0, -5),
                     xycoords="axes fraction",
                     textcoords='offset points',
                     va='center', ha="center")

    axes[4].annotate('Frac. SPCA', xy=(0., 0.3), xytext=(-10, 0),
                     xycoords="axes fraction",
                     textcoords='offset points',
                     va='center', ha="center", rotation='vertical')
    axes[0].annotate('Sparse PCA', xy=(0., 0.6), xytext=(-10,
                                                                       0.),
                     xycoords="axes fraction",
                     textcoords='offset points',
                     va='center', ha="center", rotation='vertical')

    axes[0].annotate('$x = 42$', xy=(0.5, 1), xytext=(0, 7),
                     xycoords="axes fraction",
                     textcoords='offset points',
                     va='center', ha="center")
    axes[1].annotate('$z = 9$', xy=(0.5, 1), xytext=(0, 7),
                     xycoords="axes fraction",
                     textcoords='offset points',
                     va='center', ha="center")
    axes[2].annotate('$x = 42$', xy=(0.5, 1), xytext=(0, 7),
                     xycoords="axes fraction",
                     textcoords='offset points',
                     va='center', ha="center")
    axes[3].annotate('$z = 9$', xy=(0.5, 1), xytext=(0, 7),
                     xycoords="axes fraction",
                     textcoords='offset points',
                     va='center', ha="center")
    plt.savefig(join(figures_dir, 'median.pgf'), bbox_inches="tight")
    plt.savefig(join(figures_dir, 'median.svg'), bbox_inches="tight")
    plt.savefig(join(figures_dir, 'median.pdf'), bbox_inches="tight")