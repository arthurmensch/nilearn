"""
Comparison of Dictionary Learning and ICA for doing group analysis of
resting-state fMRI
=====================================================

An example applying dictionary learning and ICA to resting-state data,
comparing resulting components using 4D plotting.

Dictionary learning is a sparsity based decomposition method for extracting
spatial maps. It extracts maps that are naturally sparse and usually cleaner
than ICA

    * Gael Varoquaux et al.
    Multi-subject dictionary learning to segment an atlas of brain spontaneous
    activity
    Information Processing in Medical Imaging, 2011, pp. 562-573, Lecture Notes
    in Computer Science

https://hal.inria.fr/inria-00588898/en/
"""

### Load ADHD rest dataset ####################################################
import datetime
import os
from os.path import expanduser, join

from nilearn import datasets
from nilearn.datasets import fetch_atlas_smith_2009
from nilearn_sandbox.datasets.fetch_adni import \
    fetch_adni_longitudinal_rs_fmri_DARTEL, fetch_adni_masks
from nilearn_sandbox.plotting.papaya import papaya_viewer

adhd_dataset = datasets.fetch_adhd(n_subjects=40)
# adni_dataset = fetch_adni_longitudinal_rs_fmri_DARTEL()
# mask = fetch_adni_masks().fmri

func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

import numpy as np

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

### Apply Decomposition estimators#############################################
from nilearn.decomposition import DictLearning, CanICA, SparsePCA

n_components = 20

### Dictionary learning #######################################################
dict_learning = DictLearning(n_components=n_components, smoothing_fwhm=6.,
                             memory="nilearn_cache", memory_level=2,
                             verbose=1,
                             alpha=2,
                             random_state=0,
                             n_epochs=1)
### CanICA ####################################################################
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=2,
                threshold=3.,
                verbose=1)


output_dir = join(expanduser('~/output/compare'),
                  datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                   '-%M-%S'))
os.makedirs(output_dir)

sparse_pca = SparsePCA(n_components=n_components, smoothing_fwhm=6.,
                       memory="nilearn_cache", memory_level=2,
                       feature_ratio=1,
                       verbose=1,
                       alpha=0.01,
                       random_state=0,
                       n_epochs=3,
                       dict_init=fetch_atlas_smith_2009().rsn20,
                       n_jobs=1,
                       debug_folder=output_dir,
                       )

### Fitting both estimators ###################################################
estimators = [sparse_pca]
components_imgs = []

for estimator in estimators:
    print('[Example] Learning maps using %s model' % type(estimator).__name__)
    estimator.fit(func_filenames)
    print('[Example] Dumping results')
    # Decomposition estimator embeds their own masker
    masker = estimator.masker_
    components_img = masker.inverse_transform(estimator.components_)
    filename = '%s_resting_state.nii.gz' % type(estimator).__name__
    components_img.to_filename(filename)
    papaya_viewer(filename, output_file=filename + 'html')
    components_imgs.append(components_img)

### Visualize the results #####################################################
# Show components from both methods using 4D plotting tools
import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, find_xyz_cut_coords
from nilearn.image import index_img

print('[Example] Displaying')

fig, axes = plt.subplots(nrows=len(estimators))
if not isinstance(axes, np.ndarray):
    axes = [axes]
# We select pertinent cut coordinates for displaying
cut_coords = find_xyz_cut_coords(index_img(components_imgs[0], 1))
for estimator, cur_img, ax in zip(estimators, components_imgs, axes):
    plot_prob_atlas(cur_img, view_type="filled_contours",
                    title="%s" % estimator.__class__.__name__,
                    axes=ax,
                    cut_coords=cut_coords, colorbar=False)
plt.show()
