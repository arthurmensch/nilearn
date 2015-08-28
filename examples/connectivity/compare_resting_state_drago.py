import glob
from os.path import join
import os
import time
import pickle
import datetime
from joblib import delayed, Parallel
import numpy as np
from nilearn.image import index_img
from nilearn.input_data import MultiNiftiMasker
from nilearn import datasets
from nilearn.decomposition import SparsePCA, DictLearning
from nilearn.decomposition.base import DecompositionEstimator
from nilearn.plotting.plot_to_pdf import plot_to_pdf
from nilearn._utils.map_alignment import align_with_last_nii

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
    size = estimator.debug_info_[0].shape[0]
    fig, axes = plt.subplots(3, 1, figsize=a4_size, sharex=True)
    titles = ['Residuals', 'Sparsity', 'Voxels trajectories']
    ylabels = ['Value', 'Value', 'Voxel value']
    for i, (ax, data) in enumerate(zip(axes, estimator.debug_info_)):
        ax.plot(data)
        ax.set_xlim(0, size)
        ax.set_title(titles[i])
        if i == 2:
            ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabels[i])
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
    estimator.fit(func_filenames)
    timing = time.time() - t0
    print('[Example] Dumping results')
    dump_debug_to_pdf(estimator, exp_output)
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_filename = join(exp_output, 'components.nii.gz')
    components_img.to_filename(components_filename)
    print('[Example] Preparing pdf')
    plot_to_pdf(components_img, path=join(exp_output, 'components.pdf'))
    return components_filename, timing


def dump_nii_and_pdf(i, components, dump_dir):
    print("[Example] Dropping aligned components % i" % i)
    filename = join(dump_dir, "experiment_%i.nii.gz" % i)
    components.to_filename(filename)
    print('[Example] Preparing pdf %i' % i)
    plot_to_pdf(components, path=join(dump_dir,
                                    "experiment_%i.pdf" % i))
    return filename


def run_experiment(n_jobs=6):
    output = os.path.expanduser('~/work/output/compare')
    output = join(output, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(join(output))
    except:
        pass

    dataset = datasets.fetch_hcp_rest(n_subjects=2, data_dir='/storage/data')
    mask = '/storage/data/HCP_mask/mask.nii.gz'
    smith = datasets.fetch_atlas_smith_2009()
    dict_init = smith.rsn20
    n_components = 20
    data_filenames = dataset.func

    print('First functional nifti image (4D) is at: %s' %
          dataset.func[0])

    # This is hacky and should be integrated in the nilearn API in a smooth way
    # Warming up cache with masked images
    print("[Example] Warming up cache")
    decomposition_estimator = DecompositionEstimator(smoothing_fwhm=4.,
                                                     memory="nilearn_cache",
                                                     mask=mask,
                                                     memory_level=3,
                                                     verbose=1,
                                                     n_jobs=n_jobs)
    decomposition_estimator.fit(data_filenames, preload=True)
    masker = decomposition_estimator.masker_

    reduction_ratios = [0.1]

    estimators = []

    # for reduction_ratio in reduction_ratios:
    #     sparse_pca = SparsePCA(n_components=n_components, mask=masker,
    #                            memory="nilearn_cache", dict_init=dict_init,
    #                            reduction_ratio=reduction_ratio,
    #                            memory_level=3,
    #                            alpha=0.1,
    #                            batch_size=20,
    #                            verbose=1,
    #                            shuffle=True,
    #                            random_state=0, l1_ratio=0.5,
    #                            n_epochs=1)
    #     estimators.append(sparse_pca)

    for reduction_ratio in reduction_ratios:
        dict_learning = DictLearning(n_components=n_components,
                                     mask=masker,
                                     memory="nilearn_cache",
                                     dict_init=dict_init,
                                     reduction_ratio=reduction_ratio,
                                     memory_level=3,
                                     batch_size=20,
                                     verbose=1,
                                     random_state=0, alpha=4, max_nbytes=0,
                                     n_epochs=1)
        estimators.append(dict_learning)

    with open(join(output, 'estimators'), mode='w+') as f:
        pickle.dump(estimators, f)

    res = Parallel(n_jobs=n_jobs, verbose=10)(delayed(fit_and_dump)(index,
                                                               estimator,
                                                               data_filenames,
                                                               output)
                                         for index, estimator
                                         in enumerate(estimators))

    components_filename, timings = zip(*res)
    timings = np.array(timings)
    np.save(join(output, 'timings'), timings)
    print(timings)

    print("Performing alignment")
    map_masker = MultiNiftiMasker(mask_img=masker.mask_img_,).fit()
    components_list = align_with_last_nii(map_masker, components_filename)
    comparison_dir = join(output, "comparison")
    os.mkdir(comparison_dir)
    Parallel(n_jobs=n_jobs)(
        delayed(dump_nii_and_pdf)(i, components, comparison_dir)
        for i, components in enumerate(components_list))

if __name__ == '__main__':
    t0 = time.time()
    run_experiment(n_jobs=6)
    time = time.time() - t0
    print('Total_time : %f s' % time)
