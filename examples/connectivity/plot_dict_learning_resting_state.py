"""
Group analysis of resting-state fMRI with dictionary learning: DictLearning
=====================================================

An example applying dictionary learning to resting-state data. This example applies it
to 10 subjects of the ADHD200 datasets.

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
from sklearn.grid_search import GridSearchCV
import os
import datetime

output_dir = os.path.expanduser('~/work/output/nilearn/plot_dict_learning_resting_state')
output_dir = os.path.join(output_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(output_dir)
except OSError:
    pass

adhd_dataset = datasets.fetch_adhd(n_subjects=10)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

### Apply DictLearning ########################################################
from nilearn.decomposition.dict_learning import DictLearning
n_components = 50

dict_learning = DictLearning(n_components=n_components, smoothing_fwhm=6.,
                             memory="nilearn_cache", memory_level=5,
                             threshold=1. * n_components, random_state=0,
                             l1_ratio=1, method='spca', verbose=10,
                             n_jobs=3, n_init=1, alpha=0.1 * n_components, n_iter=150)

# tuned_parameters = {'threshold': [1., n_components / 10., float(n_components)]}
# grid_search = GridSearchCV(dict_learning,
#                            tuned_parameters, scoring=DictLearning.score, cv=3, verbose=10)
#
dict_learning.fit(func_filenames)
#
# print(grid_search.grid_scores_)
# dict_learning = grid_search.best_estimator_

print('')
print('[Example] Dumping results')

# Retrieve learned spatial maps in brain space
components_img = dict_learning.masker_.inverse_transform(dict_learning.components_)
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
components_img.to_filename(os.path.join(output_dir, 'dict_learning_resting_state.nii.gz'))



### Visualize the results #####################################################
# Show some interesting components
print('[Example] Saving PDF')

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
from matplotlib.backends.backend_pdf import PdfPages
import nibabel
import os

map_img = nibabel.load(os.path.join(output_dir, "dict_learning_resting_state.nii.gz"))

fig = plt.figure()

with PdfPages(os.path.join(output_dir, 'output.pdf')) as pdf:
    for j in range(n_components):
        plt.clf()
        plot_stat_map(index_img(map_img, j), figure=fig, threshold="auto")
        pdf.savefig()
    plt.close()
    d = pdf.infodict()
    d['Title'] = 'Maps'