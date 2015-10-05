import collections
import os
from os.path import expanduser, join, exists
import itertools
import datetime
import shutil
import warnings
from joblib import Parallel, delayed
from nilearn_sandbox._utils.map_alignment import _align_one_to_one_flat, \
    _spatial_correlation_flat
from sklearn.utils import gen_even_slices
from theano.gradient import np
from nilearn._utils import check_niimg, copy_img
from nilearn.decomposition import SparsePCA, DictLearning
from nilearn.decomposition.base import MaskReducer, DecompositionEstimator
from nilearn import datasets
import pandas as pd
from sklearn.base import clone
import json
from nilearn.input_data import NiftiMasker, MultiNiftiMasker

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
                                     'exp_type',
                                     'n_epochs',
                                     # Out of core dictionary learning specifics
                                     'temp_dir',
                                     'reduction_ratio',
                                     'compression_type',
                                     'data',
                                     'subject_limits',
                                     # Stability specific
                                     'n_exp',
                                     'n_runs'])


def load_dataset(exp_params):
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

    print("[Experiment] Warming up cache")
    mask_reducer = MaskReducer(masker,
                               memory_level=2,
                               memory=cache_dir,
                               n_jobs=n_jobs)

    exp_type = exp_params.exp_type

    if exp_type == 'time_vs_corr':
        mask_reducer.set_params(mock=True,
                                in_memory=True)
    elif exp_type == 'out_of_core_dl':
        compression_type = exp_params.compression_type
        reduction_ratio = exp_params.reduction_ratio
        temp_dir = exp_params.temp_dir
        mask_reducer.set_params(mock=False,
                                in_memory=False,
                                compression_type=compression_type,
                                reduction_ratio=reduction_ratio,
                                temp_folder=temp_dir,
                                mem_name='concat')
    mask_reducer.fit(dataset)
    if exp_type == 'time_vs_corr':
        return dataset, masker
    if exp_type == 'out_of_core_dl':
        data = mask_reducer.data_
        subject_limits = mask_reducer.subject_limits_
        return dataset, masker, data, subject_limits


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
    for i, estimator in enumerate(estimators):
        reference = (i == len(estimators) - 1)
        offset = 100 if reference else 0
        for random_state in offset + np.arange(n_runs):
            estimator = clone(estimator)
            estimator.set_params(mask=masker,
                                 smoothing_fwhm=smoothing_fwhm,
                                 n_epochs=n_epochs,
                                 n_jobs=1,
                                 dict_init=dict_init,
                                 n_components=n_components,
                                 memory_level=2, memory=cache_dir,
                                 verbose=3,
                                 random_state=random_state)
            yield estimator, (i == len(estimators) - 1)


def single_run(index, estimator, dataset, output_dir, reference):
    exp_output = join(output_dir, "experiment_%i" % index)
    os.mkdir(exp_output)
    if type(estimator).__name__:
        debug_folder = join(exp_output, 'debug')
        os.mkdir(debug_folder)
    estimator.set_params(debug_folder=debug_folder)
    print('[Example] Learning maps')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        estimator.fit(dataset)
    print('[Example] Dumping results')
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_filename = join(exp_output, 'components.nii.gz')
    components_img.to_filename(components_filename)
    single_run_dict = {'estimator_type': type(estimator).__name__,
                       'compression_type': estimator.compression_type,
                       'reduction_ratio':
                           estimator.reduction_ratio,
                       'alpha': estimator.alpha,
                       'random_state': estimator.random_state,
                       # Columns
                       'components': components_filename,
                       'math_time': estimator.time_[0],
                       'io_time': estimator.time_[1],
                       'reference': reference
                       }
    print(single_run_dict)
    return single_run_dict


def run(estimators, exp_params):
    output_dir = join(exp_params.output_dir,
                      datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                       '-%M-%S'))
    os.mkdir(output_dir)
    with open(join(output_dir, 'experiment.json'), 'w+') as f:
        json.dump(exp_params.__dict__, f)

    dataset, masker = load_dataset(exp_params)

    dataset_series = pd.Series(dataset)
    dataset_series.to_csv(join(output_dir, 'dataset.csv'))

    dict_init, n_components = check_init(exp_params)

    check_niimg(dict_init).to_filename(join(output_dir,
                                            'dict_init.nii.gz'))
    full_dict_list = Parallel(n_jobs=exp_params.n_jobs)(delayed(single_run)(index, estimator, dataset, output_dir,
                                                                            reference)
                                                        for index, (estimator, reference) in
                                                        enumerate(yield_estimators(estimators,
                                                                                   exp_params,
                                                                                   masker,
                                                                                   dict_init,
                                                                                   n_components)))
    results = pd.DataFrame(full_dict_list, columns=['estimator_type',
                                                    'compression_type',
                                                    'reduction_ratio',
                                                    'alpha',
                                                    'random_state',
                                                    'math_time', 'io_time',
                                                    'reference',
                                                    'components'])

    results.sort_index(by=['estimator_type',
                           'compression_type',
                           'reduction_ratio',
                           'alpha',
                           'random_state'], inplace=True)
    results.to_csv(join(output_dir, 'results.csv'))
    copy_img(masker.mask_img_).to_filename(join(output_dir,
                                                'mask_img.nii.gz'))
    return output_dir


def align_single(masker, stack_base, results_dir, exp_int_index, index, sub_df):
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
    print('[Experiment] Performing Hungarian alg. and computing correlation score')

    stack_base = np.concatenate(masker.transform(results.loc[results['reference'], 'components']))
    masker.inverse_transform(stack_base).to_filename(join(results_dir, 'base.nii.gz'))
    res_list = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(align_single)(masker, stack_base, results_dir, exp_int_index, index, sub_df)
        for exp_int_index, (index, sub_df) in enumerate(results.groupby(
            level=['estimator_type', 'compression_type', 'reduction_ratio', 'alpha'])))
    for index, score, aligned_filename in res_list:
        results.loc[index, 'score'] = score
        results.loc[index, 'aligned_filename'] = aligned_filename

    time_v_corr = results.drop('components', axis='columns')
    time_v_corr.reset_index(level='random_state', drop=True, inplace=True)

    # Selection best scoring alpha for each parameter set
    indices = time_v_corr.groupby(level=['estimator_type',
                                         'compression_type',
                                         'reduction_ratio']).apply(lambda x: x['score'].idxmax())
    time_v_corr = time_v_corr.loc[indices]
    time_v_corr.reset_index(level='alpha', drop=False, inplace=True)
    # Mean over random_state
    time_v_corr = time_v_corr.groupby(level=['estimator_type',
                                             'compression_type',
                                             'reduction_ratio']).agg({'math_time': np.mean,
                                                                      'io_time': np.mean,
                                                                      'alpha': 'last',
                                                                      'reference': 'last',
                                                                      'aligned_filename': 'last',
                                                                      'score': 'last'})

    time_v_corr.to_csv(join(results_dir, 'time_v_corr.csv'))


def align_incr_single(masker, base_list, n_exp, index, sub_df):
    target_list = masker.transform(sub_df['components'])
    base = np.concatenate(base_list[:(n_exp + 1)])
    target = np.concatenate(target_list[:(n_exp + 1)])
    aligned = _align_one_to_one_flat(base, target)
    return index, n_exp, np.trace(_spatial_correlation_flat(aligned, base)) / len(base)


def analyse_incr(output_dir, n_jobs=1):
    results_dir = join(output_dir, 'stability')
    results = pd.read_csv(join(output_dir, 'results.csv'), index_col=0)
    results.set_index(['estimator_type', 'compression_type', 'reduction_ratio',
                       'alpha', 'random_state'], inplace=True)
    results.reset_index(inplace=True)
    results.set_index(['estimator_type', 'compression_type', 'reduction_ratio', 'alpha'], inplace=True)

    time_v_corr = pd.read_csv(join(results_dir, 'time_v_corr.csv'), index_col=0)
    time_v_corr.reset_index(inplace=True)
    time_v_corr.set_index(['estimator_type', 'compression_type', 'reduction_ratio', 'alpha'], inplace=True)

    mask = check_niimg(join(output_dir, 'mask_img.nii.gz'))
    masker = MultiNiftiMasker(mask_img=mask).fit()

    base_list = masker.transform(results.loc[results['reference'], 'components'])

    joined_results = results.join(time_v_corr, how='inner', rsuffix='_mean')
    joined_results.reset_index(inplace=True)
    joined_results.set_index(['estimator_type', 'compression_type', 'reduction_ratio',
                              'random_state'], inplace=True)

    incr_df = pd.DataFrame(columns=np.arange(len(base_list)), index=joined_results.index)
    res = Parallel(n_jobs=n_jobs, verbose=3)(delayed(align_incr_single)(masker, base_list, n_exp, index, sub_df)
                                             for index, sub_df in joined_results.groupby(level=['estimator_type',
                                                                                                'compression_type',
                                                                                                'reduction_ratio'])
                                             for n_exp in range(len(base_list)))
    for index, n_exp, score in res:
        incr_df.loc[index, n_exp] = score
    incr_df = incr_df.groupby(level=['estimator_type', 'compression_type', 'reduction_ratio']).last()
    time_v_corr.reset_index(inplace=True)
    time_v_corr.set_index(['estimator_type', 'compression_type', 'reduction_ratio'], inplace=True)
    stability = pd.concat({'score': time_v_corr, 'incr': incr_df}, axis=1)
    stability.to_csv(join(results_dir, 'stability.csv'))


estimators = []
alpha_list = {'range_finder': [16, 16, 16, 16, 14],
              'subsample': [10, 10, 10, 12, 14]}

shutil.rmtree(expanduser('~/nilearn_cache/joblib/sklearn'))
shutil.rmtree(expanduser('~/nilearn_cache/joblib/scipy'))

for compression_type in ['range_finder', 'subsample']:
    for reduction_ratio in np.linspace(0.2, 1, 5):
        for alpha in alpha_list[compression_type]:
            estimators.append(DictLearning(alpha=alpha, batch_size=20,
                                           compression_type=compression_type,
                                           random_state=0,
                                           forget_rate=1,
                                           reduction_ratio=reduction_ratio))
# Baseline
estimators.append(DictLearning(alpha=20, batch_size=20,
                               compression_type='none',
                               random_state=0,
                               forget_rate=1,
                               reduction_ratio=1))
experiment = Experiment('adhd',
                        n_subjects=40,
                        smoothing_fwhm=6,
                        dict_init='rsn20',
                        output_dir=expanduser('~/output'),
                        cache_dir=expanduser('~/nilearn_cache'),
                        data_dir=expanduser('~/data'),
                        n_slices=1,
                        n_jobs=30,
                        exp_type='time_vs_corr',
                        n_epochs=1,
                        # Out of core dictionary learning specifics
                        temp_dir=expanduser('~/temp'),
                        reduction_ratio=None,
                        compression_type=None,
                        data=None,
                        subject_limits=None,
                        # Stability specific
                        n_exp=None,
                        n_runs=10)

output_dir = run(estimators, experiment)
# analyse('/volatile/arthur/work/output/2015-10-05_12-48-04', n_jobs=10)
# analyse_incr('/volatile/arthur/work/output/2015-10-05_12-48-04', n_jobs=10)
