import glob
from os.path import join
import os
import time
import pickle
import datetime
import warnings
from joblib import delayed, Parallel
import numpy as np

import matplotlib
from nilearn._utils import check_niimg

matplotlib.use('PDF')

from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn import datasets
from nilearn.decomposition import SparsePCA, DictLearning, CanICA
from nilearn.decomposition.base import DecompositionEstimator
from nilearn_sandbox.plotting.pdf_plotting import plot_to_pdf
from nilearn_sandbox._utils.map_alignment import align_list_with_last_nii, \
    spatial_correlation

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nilearn.plotting import plot_prob_atlas, plot_stat_map


def compare(x, y):
    if len(x) < len(y):
        return -1
    elif len(x) == len(y):
        return cmp(x, y)
    else:
        return 1


def dump_debug_to_pdf(estimator, output):
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
        f.write("Math %f - IO %f" % (estimator.time_[0], estimator.time_[1]))
        f.write('\n')
    evolution = sorted(glob.glob(join(output, 'debug', 'components_*.nii.gz')),
                       compare)
    with PdfPages(join(output, 'evolution.pdf')) as pdf:
        for i in range(0, len(evolution), 5):
            fig, axes = plt.subplots(5, 1, figsize=a4_size, squeeze=False)
            axes = axes.reshape(-1)
            for j, ax in enumerate(axes):
                if i + j < len(evolution):
                    plot_prob_atlas(evolution[i + j], axes=ax)
                else:
                    ax.axis('off')
            pdf.savefig(fig)
            plt.close()

    with PdfPages(join(output, 'evolution_single.pdf')) as pdf:
        for i in range(0, len(evolution), 5):
            fig, axes = plt.subplots(5, 1, figsize=a4_size, squeeze=False)
            axes = axes.reshape(-1)
            for j, ax in enumerate(axes):
                if i + j < len(evolution):
                    plot_stat_map(index_img(evolution[i + j], 0), axes=ax)
                else:
                    ax.axis('off')
            pdf.savefig(fig)
            plt.close()


def fit_and_dump(index, estimator, func_filenames, output):
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
    dump_debug_to_pdf(estimator, exp_output)
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_filename = join(exp_output, 'components.nii.gz')
    components_img.to_filename(components_filename)
    print('[Example] Preparing pdf')
    plot_to_pdf(components_img, path=join(exp_output, 'components.pdf'))
    timing = np.zeros(3)
    timing[0:2] = estimator.time_
    timing[2] = full_time
    return components_filename, timing


def dump_nii_and_pdf(i, masker, components, reference, dump_dir):
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
    return np.mean(np.diag(corr))


def run_experiment(estimators, init='rsn70', n_epochs=1,
                   dataset='adhd',
                   n_subjects=40,
                   smoothing_fwhm=4.,
                   n_jobs=6, parallel_exp=True):
    output = os.path.expanduser('~/work/output/compare')
    temp_dir = os.path.expanduser('~/temp')
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
        mask = None  # os.path.expanduser('~/data/ADHD_mask/mask_img.nii.gz')
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
    decomposition_estimator.fit(data_filenames, preload=True,
                                temp_dir=temp_dir)
    masker = decomposition_estimator.masker_

    estimator_n_jobs = n_jobs if not parallel_exp else 1

    for estimator in estimators:
        # Setting technical parameters
        estimator.set_params(mask=masker, dict_init=dict_init,
                             smoothing_fwhm=smoothing_fwhm,
                             n_components=n_components,
                             n_epochs=n_epochs,
                             n_jobs=estimator_n_jobs,
                             random_state=0,
                             memory_level=2, memory='nilearn_cache',
                             verbose=3)

    result_dict = []
    with open(join(output, 'estimators'), 'w+') as f:
        for estimator in estimators:
            f.write("%s\n" % estimator)
            result_dict.append(dict(type=type(estimator).__name__,
                                    alpha=estimator.alpha,
                                    batch_size=estimator.batch_size,
                                    compression_type=
                                    estimator.compression_type,
                                    reduction_ratio=
                                    estimator.reduction_ratio,
                                    power_iter=estimator.power_iter,
                                    feature_compression=estimator.
                                    feature_compression,
                                    forget_rate=estimator.forget_rate))

    exp_n_jobs = n_jobs if parallel_exp else 1

    res = Parallel(n_jobs=exp_n_jobs, verbose=10)\
        (delayed(fit_and_dump)(index,
                               estimator,
                               data_filenames,
                               output)
         for index, estimator
         in enumerate(estimators))

    components_filename, timings_list = zip(*res)
    timings = np.zeros((len(estimators), 3))
    for i, timing in enumerate(timings_list):
        timings[i] = np.array(timing)
        result_dict[i]['math_time'] = timings[i, 0]
        result_dict[i]['io_time'] = timings[i, 1]
        result_dict[i]['total_time'] = timings[i, 2]
    np.save(join(output, 'timings'), timings)

    masker.mask_img_.to_filename(join(output, 'mask_img.nii.gz'))
    
    if len(estimators) > 1:
        print("Performing alignment")
        map_masker = MultiNiftiMasker(mask_img=masker.mask_img_,).fit()
        components_list = align_list_with_last_nii(map_masker,
                                                   components_filename)
        comparison_dir = join(output, "comparison")
        os.mkdir(comparison_dir)
        corr_list = Parallel(n_jobs=n_jobs)(
            delayed(dump_nii_and_pdf)(i, map_masker, components,
                                      components_list[-1], comparison_dir)
            for i, components in enumerate(components_list))
        np.save('correlations', np.array(corr_list))
        for i, score in enumerate(corr_list):
            result_dict[i]['score'] = score
    pickle.dump(result_dict, open(join(output, 'results'), 'w+'))
    with open(join(output, 'results.txt'), 'w+') as f:
        f.write("%s" % result_dict)
#
#
# def run_stability(estimator, slices, init='rsn70', n_epochs=1,
#                   dataset='adhd',
#                   n_subjects=40,
#                   smoothing_fwhm=4.,
#                   n_jobs=6, parallel_exp=True):
#     output = os.path.expanduser('~/work/output/compare')
#     temp_dir = os.path.expanduser('~/temp')
#     cache_dir = os.path.expanduser('~/nilearn_cache')
#     data_dir = os.path.expanduser('~/data')
#     output = join(output, datetime.datetime.now().strftime('%Y-%m-%d_%H'
#                                                            '-%M-%S'))
#     try:
#         os.makedirs(output)
#     except:
#         pass
#
#     if dataset == 'adhd':
#         dataset = datasets.fetch_adhd(n_subjects=min(40, n_subjects))
#         mask = None  # os.path.expanduser('~/data/ADHD_mask/mask_img.nii.gz')
#     elif dataset == 'hcp':
#         dataset = datasets.fetch_hcp_rest(n_subjects=n_subjects,
#                                           data_dir=data_dir)
#         mask = os.path.expanduser('~/data/HCP_mask/mask_img.nii.gz')
#     smith = datasets.fetch_atlas_smith_2009()
#     if isinstance(init, int):
#         dict_init = None
#         n_components = init
#     elif init == 'rsn70':
#         dict_init = smith.rsn70
#         n_components = 70
#     elif init == 'rsn20':
#         dict_init = smith.rsn20
#         n_components = 20
#     else:
#         if os.path.exists(init):
#             dict_init = init
#             n_components = check_niimg(init).get_shape()[3]
#         else:
#             raise ValueError('Unsupported init')
#     data_filenames = dataset.func
#
#     print('First functional nifti image (4D) is at: %s' %
#           dataset.func[0])
#
#     # This is hacky and should be integrated in the nilearn API in a smooth way
#     # Warming up cache with masked images
#     print("[Example] Warming up cache")
#     decomposition_estimator = DecompositionEstimator(smoothing_fwhm=
#                                                      smoothing_fwhm,
#                                                      memory=cache_dir,
#                                                      mask=mask,
#                                                      memory_level=2,
#                                                      verbose=10,
#                                                      n_jobs=n_jobs)
#     decomposition_estimator.fit(data_filenames, preload=True,
#                                 temp_dir=temp_dir)
#     masker = decomposition_estimator.masker_
#
#     estimator_n_jobs = n_jobs if not parallel_exp else 1
#
#     # Setting technical parameters
#     estimator.set_params(mask=masker, dict_init=dict_init,
#                          smoothing_fwhm=smoothing_fwhm,
#                          n_components=n_components,
#                          n_epochs=n_epochs,
#                          n_jobs=estimator_n_jobs,
#                          random_state=0,
#                          memory_level=2, memory='nilearn_cache',
#                          verbose=3)
#
#     with open(join(output, 'estimators'), 'w+') as f:
#         f.write("%s\n" % estimator)
#
#     exp_n_jobs = n_jobs if parallel_exp else 1
#
#     res = Parallel(n_jobs=exp_n_jobs, verbose=10)(
#         delayed(fit_and_dump)(index,
#                               estimator,
#                               data_filenames[this_slice],
#                               output)
#          for index, this_slice
#          in enumerate(slices))
#
#     components_filename, timings_list = zip(*res)
#     timings = np.zeros((len(timings_list), 3))
#     for i, timing in enumerate(timings_list):
#         timings[i] = np.array(timing)
#     np.save(join(output, 'timings'), timings)
#     masker.mask_img_.to_filename(join(output, 'mask_img.nii.gz'))
#
#     if len(timings_list) > 1:
#         print("Performing alignment")
#         map_masker = MultiNiftiMasker(mask_img=masker.mask_img_,).fit()
#         components_list = align_list_with_last_nii(map_masker,
#                                                    components_filename)
#         comparison_dir = join(output, "comparison")
#         os.mkdir(comparison_dir)
#         Parallel(n_jobs=n_jobs)(
#             delayed(dump_nii_and_pdf)(i, components, comparison_dir)
#             for i, components in enumerate(components_list))


if __name__ == '__main__':
    t0 = time.time()

    estimators = []
    # for reduction_ratio in [0.1, 0.5, 1]:
    #     estimators.append(DictLearning(alpha=10, batch_size=40,
    #                                    compression_type='subsample',
    #                                    forget_rate=1,
    #                                    reduction_ratio=0.1))
    for compression_type in ['subsample', 'range_finder']:
        for reduction_ratio in [0.1, 0.25, 0.5]:
            for alpha in [5, 7, 9, 11, 13]:
                estimators.append(DictLearning(alpha=10, batch_size=20,
                                               compression_type='compression_type',
                                               forget_rate=1,
                                               reduction_ratio=reduction_ratio,
                                               feature_compression=1))

    for feature_compression in [0.1, 0.25, 0.5]:
        estimators.append(DictLearning(alpha=15, batch_size=20,
                                       compression_type='none',
                                       forget_rate=1,
                                       reduction_ratio=1,
                                       feature_compression=feature_compression)
                          )

    # Baseline
    estimators.append(DictLearning(alpha=15, batch_size=20,
                                   compression_type='subsample',
                                   forget_rate=1,
                                   reduction_ratio=1))

    run_experiment(estimators, n_jobs=8, dataset='adhd', n_subjects=40,
                   smoothing_fwhm=6.,
                   init="rsn70",
                   n_epochs=1)
    time = time.time() - t0
    print('Total_time : %f s' % time)
