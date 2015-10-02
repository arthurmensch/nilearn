import collections
import os
from os.path import expanduser, join, exists
import itertools
import datetime
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
    exp_type = exp_params.exp_type
    n_runs = exp_params.n_runs
    cache_dir = exp_params.cache_dir
    for estimator in estimators:
        for random_state in range(n_runs):
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
            yield estimator


def single_run(index, estimator, dataset, output_dir):
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
                       'io_time': estimator.time_[1]
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
    full_dict_list = Parallel(n_jobs=exp_params.n_jobs)(delayed(single_run)(index, estimator, dataset, output_dir)
                                                        for index, estimator in
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


def analyse_inc(output_dir):
    return


def analyse(output_dir):
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

    base_index = results.reset_index('random_state').index[-1]
    stack_base = np.concatenate(masker.transform(results.loc[base_index, 'components']))
    masker.inverse_transform(stack_base).to_filename(join(results_dir, 'base.nii.gz'))

    for i, (index, sub_df) in enumerate(results.groupby(level=['estimator_type',
                                                               'compression_type',
                                                               'reduction_ratio',
                                                               'alpha'])):
        stack_target = np.concatenate(masker.transform(sub_df['components']))
        aligned = _align_one_to_one_flat(stack_base, stack_target)
        filename = join(results_dir, 'aligned_%i.nii.gz' % i)
        masker.inverse_transform(aligned).to_filename(filename)
        corr = _spatial_correlation_flat(aligned, stack_base)
        results.loc[index, 'score'] = np.mean(corr.diagonal())
        results.loc[index, 'aligned_filename'] = filename

    time_v_corr = results.drop('components', axis='columns')
    time_v_corr.reset_index(level='random_state', drop=True, inplace=True)

    # Selection best scoring alpha for each parameter set
    indices = time_v_corr.groupby(level=['estimator_type',
                                 'compression_type',
                                 'reduction_ratio']).apply(lambda x : x['score'].idxmax())
    time_v_corr = time_v_corr.loc[indices]

    # Mean over random_state
    time_v_corr = time_v_corr.groupby(level=['estimator_type',
                                 'compression_type',
                                 'reduction_ratio']).mean()

    time_v_corr['filename'] = filename
    time_v_corr.to_csv(join(results_dir, 'time_v_corr.csv'))



estimators = []
for compression_type in ['subsample', 'range_finder']:
    for reduction_ratio in np.linspace(0.2, 1, 5):
        for alpha in np.linspace(10, 20, 6):
            estimators.append(DictLearning(alpha=alpha, batch_size=20,
                                        compression_type=compression_type,
                                        random_state=0,
                                        forget_rate=1,
                                        reduction_ratio=reduction_ratio))
experiment = Experiment('adhd',
                        n_subjects=40,
                        smoothing_fwhm=4,
                        dict_init='rsn20',
                        output_dir=expanduser('~/output'),
                        cache_dir=expanduser('~/nilearn_cache'),
                        data_dir=expanduser('~/data'),
                        n_slices=1,
                        n_jobs=16,
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
                        n_runs=3)

# output_dir = run(estimators, experiment)
analyse('/volatile/arthur/work/output/2015-10-02_14-52-07')