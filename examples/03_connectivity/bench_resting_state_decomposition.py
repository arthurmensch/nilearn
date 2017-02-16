"""
Dictionary Learning and ICA for doing group analysis of resting-state fMRI
==========================================================================

This example applies dictionary learning and ICA to resting-state data,
visualizing resulting components using atlas plotting tools.

Dictionary learning is a sparsity based decomposition method for extracting
spatial maps. It extracts maps that are naturally sparse and usually cleaner
than ICA

    * Gael Varoquaux et al.
      Multi-subject dictionary learning to segment an atlas of brain spontaneous
      activity
      Information Processing in Medical Imaging, 2011, pp. 562-573, Lecture Notes
      in Computer Science

Available on https://hal.inria.fr/inria-00588898/en/
"""
###############################################################################
# Load ADHD rest dataset
from os.path import expanduser

from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects=40)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

###############################################################################
# Create two decomposition estimators
from nilearn.decomposition import DictLearning

n_components = 200

###############################################################################
# Dictionary learning
dict_learning = DictLearning(n_components=n_components,
                             memory=expanduser('~/cache'), memory_level=2,
                             verbose=1,
                             random_state=0,
                             screening=0,
                             n_epochs=1)
###############################################################################
# CanICA
dict_learning_screening = DictLearning(n_components=n_components,
                                       memory=expanduser('~/cache'), memory_level=2,
                                       verbose=1,
                                       random_state=0,
                                       screening=5,
                                       n_epochs=1)

###############################################################################
# Fit both estimators
estimators = [dict_learning, dict_learning_screening]
names = {dict_learning: 'DL',
         dict_learning_screening: 'DL_screening'}
components_imgs = []


for estimator in estimators:
    print('[Example] Learning maps using %s model' % names[estimator])
    estimator.fit(func_filenames)
    print('DL time %.4f' % estimator.dict_learning_time_)
    print('Lasso time %.4f' % estimator.sparse_encode_time_)
    print('[Example] Saving results')
    # Decomposition estimator embeds their own masker
    masker = estimator.masker_
    # Drop output maps to a Nifti   file
    components_img = masker.inverse_transform(estimator.components_)
    components_img.to_filename('%s_resting_state.nii.gz' %
                               names[estimator])
    components_imgs.append(components_img)

###############################################################################
# Visualize the results
from nilearn.plotting import (plot_prob_atlas, find_xyz_cut_coords, show,
                              plot_stat_map)
from nilearn.image import index_img

# Selecting specific maps to display: maps were manually chosen to be similar
indices = {dict_learning: 1, dict_learning_screening: 1}
# We select relevant cut coordinates for displaying
cut_component = index_img(components_imgs[0], indices[dict_learning])
cut_coords = find_xyz_cut_coords(cut_component)
for estimator, components in zip(estimators, components_imgs):
    # 4D plotting
    plot_prob_atlas(components, view_type="filled_contours",
                    title="%s" % names[estimator],
                    cut_coords=cut_coords, colorbar=False)
    # 3D plotting
    plot_stat_map(index_img(components, indices[estimator]),
                  title="%s" % names[estimator],
                  cut_coords=cut_coords, colorbar=False)
show()
