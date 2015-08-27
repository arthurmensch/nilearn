from os.path import join
import os
import time
import pickle
import datetime
import numpy as np
from nilearn import datasets
from nilearn.decomposition import SparsePCA, DictLearning
from nilearn.plotting.plot_to_pdf import plot_to_pdf

output = '/volatile/arthur/work/output/compare'
output = join(output, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(join(output))
except:
    pass

dataset = datasets.fetch_adhd(n_subjects=10)
smith = datasets.fetch_atlas_smith_2009()
dict_init = smith.rsn20
n_components = 20
func_filenames = dataset.func

reduction_ratios = [0.25, 0.5, 1]

print('First functional nifti image (4D) is at: %s' %
      dataset.func[0])
estimators = []
for reduction_ratio in reduction_ratios:
    sparse_pca = SparsePCA(n_components=n_components, smoothing_fwhm=4.,
                           memory="nilearn_cache", dict_init=dict_init,
                           reduction_ratio=reduction_ratio,
                           memory_level=3,
                           batch_size=20,
                           verbose=1,
                           random_state=0, l1_ratio=0.3,
                           n_epochs=1)
    estimators.append(sparse_pca)
for reduction_ratio in reduction_ratios:
    dict_learning = DictLearning(n_components=n_components,
                                 smoothing_fwhm=4.,
                                 memory="nilearn_cache", dict_init=dict_init,
                                 reduction_ratio=reduction_ratio,
                                 memory_level=3,
                                 batch_size=20,
                                 verbose=1,
                                 random_state=0, alpha=3.5, max_nbytes=0,
                                 n_epochs=1)
    estimators.append(dict_learning)

with open(join(output, 'estimators'), mode='w+') as f:
    pickle.dump(estimators, f)

components_imgs = []
timings = np.zeros(len(estimators))
for i, estimator in enumerate(estimators):
    exp_output = join(output, "experiment_%i" % i)
    os.mkdir(exp_output)
    if type(estimator).__name__:
        debug_folder = join(exp_output, 'debug')
        os.mkdir(debug_folder)
    estimator.debug_folder = debug_folder
    print('[Example] Learning maps using %s model' % type(estimator).__name__)
    t0 = time.time()
    estimator.fit(func_filenames)
    timings[i] = time.time() - t0
    print('[Example] Dumping results')
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_filename = join(exp_output, 'components.nii.gz')
    components_img.to_filename(components_filename)
    print('[Example] Preparing pdf')
    plot_to_pdf(components_img, path=join(exp_output, 'components.pdf'))
    components_imgs.append(components_filename)

np.save(join(output, 'timings'), timings)

import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, find_xyz_cut_coords
from nilearn.image import index_img

print('[Example] Displaying')

fig, axes = plt.subplots(nrows=len(estimators), squeeze=False)

axes = axes.reshape(-1)
cut_coords = find_xyz_cut_coords(index_img(components_imgs[0], 1))
for estimator, cur_img, timing, ax in zip(estimators, components_imgs,
                                        timings, axes):
    print('Run time of %s : %f' % (estimator.__class__.__name__, timing))
    plot_prob_atlas(cur_img, view_type="continuous",
                    title="%s" % estimator.__class__.__name__,
                    axes=ax,
                    cut_coords=cut_coords, colorbar=False)
plt.show()

