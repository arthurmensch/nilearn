"""
Group analysis of resting-state fMRI with dictionary learning: DictLearning
=====================================================

An example applying dictionary learning to resting-state data. This example applies it
to 40 subjects of the ADHD200 datasets.

Dictionary learning is a sparsity based decomposition method for extracting spatial maps.

    * Gael Varoquaux et al.
    Multi-subject dictionary learning to segment an atlas of brain spontaneous activity
    Information Processing in Medical Imaging, 2011, pp. 562-573, Lecture Notes in Computer Science

Pre-prints for paper is available on hal
(http://hal.archives-ouvertes.fr)
"""

import numpy as np

### Load ADHD rest dataset ####################################################
from nilearn import datasets

# OUTPUT DIR
import os
import datetime

output_dir = os.path.expanduser('~/work/output/nilearn/plot_dict_learning_resting_state')
output_dir = os.path.join(output_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(output_dir)
except OSError:
    pass

data_dir = '/storage/data/HCP'
func_filenames = []
print('Scanning tree')
for subdir in ['S500-1', 'S500-2', 'S500-3', 'S500-4']:
    fulldir = os.path.join(data_dir, subdir)
    for subject_id in os.listdir(fulldir):
        subject_dir = os.path.join(fulldir, subject_id, 'MNINonLinear', 'Results')
        for record_id in ['rfMRI_REST1_RL', 'rfMRI_REST1_LR', 'rfMRI_REST2_RL', 'rfMRI_REST2_LR']:
            record_dir = os.path.join(subject_dir, record_id)
            filename = os.path.join(record_dir, record_id + '.nii.gz')
            if os.path.exists(filename):
                func_filenames.append(filename)
print('Done')

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
       func_filenames[0])  # 4D data

### Apply DictLearning ########################################################
from nilearn.decomposition.dict_learning import DictLearning
from nilearn.decomposition.canica import CanICA

n_components = 50
# dict_learning = CanICA(n_components=n_components, smoothing_fwhm=6.,
#                        memory="nilearn_cache", memory_level=5,
#                        threshold=0.5, verbose=10, random_state=0,
#                        n_jobs=1, n_init=5)

dict_learning = DictLearning(mask="/home/parietal/amensch/HCP/mask_img.nii.gz", n_components=n_components,
                             smoothing_fwhm=2.,
                             memory="nilearn_cache", memory_level=5, method='enet',
                             threshold=1., verbose=10, random_state=0,
                             n_jobs=5, n_init=5, l1_ratio=0.3, alpha=3.7, n_iter=1000)

dict_learning.incremental_fit([func_filenames[-0], func_filenames[-10], func_filenames[-20],
                              func_filenames[-30], func_filenames[-40]])
dict_learning.incremental_fit(func_filenames[0:10])
# Retrieve the independent components in brain space
components_img = dict_learning.masker_.inverse_transform(dict_learning.components_)
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
components_img.to_filename(os.path.join(output_dir, 'dict_learning_resting_state.nii.gz'))

# Debug info drop
np.save(os.path.join(output_dir, 'values'), dict_learning.values_)
np.save(os.path.join(output_dir, 'residuals'), dict_learning.residuals_)
np.save(os.path.join(output_dir, 'density'), dict_learning.density_)

### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import iter_img

for i, cur_img in enumerate(iter_img(components_img)):
    if i > 10:
        break
    plot_stat_map(cur_img, display_mode="z", title="Map %d" % i, cut_coords=1,
                  colorbar=True)

plt.show()
