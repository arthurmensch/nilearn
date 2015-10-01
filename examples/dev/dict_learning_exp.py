import collections
import os
from os.path import expanduser, join, exists
import itertools
import datetime
import warnings
from nilearn_sandbox._utils.map_alignment import _align_one_to_one_flat, \
    _spatial_correlation_flat
from sklearn.utils import gen_even_slices
from theano.gradient import np
from nilearn._utils import check_niimg, copy_img
from nilearn.decomposition import SparsePCA
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
                                     'n_run'])

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
                                compression_type=
                                compression_type,
                                reduction_ratio=
                                reduction_ratio,
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

    for estimator in estimators:
        estimator = clone(estimator)
        estimator.set_params(mask=masker,
                             smoothing_fwhm=smoothing_fwhm,
                             n_epochs=n_epochs,
                             n_jobs=1,
                             dict_init=dict_init,
                             n_components=n_components,
                             memory_level=2, memory='nilearn_cache',
                             verbose=3)
        if exp_type == 'time_vs_corr':
            yield estimator
        elif exp_type == 'stability':
            for random_state in range(10):
                estimator.set_params(random_state=random_state)
                yield estimator


def single_run(estimator, dataset, output_dir, index):
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
    single_run_dict = {'components': components_filename,
                       'math_time': estimator.time_[0],
                       'io_time': estimator.time_[1]
                       }
    return single_run_dict


def run(estimators, exp_params):
    output_dir = join(exp_params.output_dir,
                      datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                       '-%M-%S'))
    os.mkdir(output_dir)
    with open(join(output_dir, 'experiment.json'), 'w+') as f:
        json.dump(exp_params.__dict__, f)

    dataset, masker = load_dataset(exp_params)

    copy_img(masker.mask_img_).to_filename(join(output_dir,
                                           'mask_img.nii.gz'))

    dataset_series = pd.Series(dataset)
    dataset_series.to_csv(join(output_dir, 'dataset.csv'))

    dict_init, n_components = check_init(exp_params)

    check_niimg(dict_init).to_filename(join(output_dir,
                                            'dict_init.nii.gz'))

    results = pd.DataFrame(columns=['estimator_type',
                                          'compression_type',
                                          'reduction_ratio',
                                          'alpha',
                                          'random_state',
                                          'math_time', 'io_time',
                                          'components'])
    for estimator in yield_estimators(estimators, exp_params, masker,
                                      dict_init,
                                      n_components):
        index = results.shape[0]
        single_run_dict = single_run(estimator, dataset,
                                     output_dir, index)
        index_dict = {'estimator_type': type(estimator).__name__,
                      'compression_type': estimator.compression_type,
                      'estimator.reduction_ratio':
                          estimator.reduction_ratio,
                      'alpha': estimator.alpha,
                      'random_state': estimator.random_state}
        full_dict = dict(single_run_dict, **index_dict)
        results.loc[results.shape[0] + 1] = pd.Series(full_dict)
        if not exists(join(output_dir, 'results.csv')):
            results.to_csv(join(output_dir, 'results.csv'))
        else:
            with open(join(output_dir, 'results.csv'), 'a') as f:
                results[-1].to_csv(f, header=False)
    return output_dir


def analyse(output_dir):
    results_dir = join(output_dir, 'stability')
    os.mkdir(results_dir)
    results = pd.from_csv(join(output_dir, 'results.csv'))
    mask = check_niimg(join(output_dir, 'mask_img.nii.gz'))
    masker = MultiNiftiMasker(mask_img=mask).fit()
    results.set_index(['estimator_type', 'compression_type', 'reduction_ratio',
                       'random_state'])
    # Concatenate over random states
    components_df = pd.DataFrame(index=['estimator_type',
                                        'compression_type',
                                        'reduction_ratio',
                                        'alpha'],
                                 columns=['components', 'score', 'math_time'])
    for params, sub_df in results.group_by(['estimator_type',
                                            'compression_type',
                                            'reduction_ratio',
                                            'alpha']):
        components_df.loc[len(components_df) + 1,
                          'components'] = masker.transform(
            sub_df['components'])
    base = components_df.loc[-1, 'components']
    for index, components in components_df['components'].iteritems():
        aligned = _align_one_to_one_flat(base, components)
        corr = _spatial_correlation_flat(aligned, base)
        components_df.loc[index, 'score'] = np.mean(corr.diagonal())
    idx = components_df.groupby(['estimator_type',
                                 'compression_type',
                                 'reduction_ratio'])['score'].idxmax()
    components_df = components_df.iloc[idx]
    for idx in components_df.index:
        components_df.loc[idx, 'math_time'] = results.loc[idx,
                                                          'math_time'].mean()
    components_df.reset_index(level=3)



estimators = []
estimators.append(SparsePCA(alpha=0.1, batch_size=20,
                            compression_type=
                            'none',
                            random_state=0,
                            forget_rate=1,
                            reduction_ratio=1))
experiment = Experiment('adhd',
                        n_subjects=2,
                        smoothing_fwhm=4,
                        dict_init='rsn20',
                        output_dir=expanduser('~/output'),
                        cache_dir=expanduser('~/cache'),
                        data_dir=expanduser('~/data'),
                        n_slices=1,
                        n_jobs=1,
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
                        n_run=None)
output_dir = run(estimators, experiment)

