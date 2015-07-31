"""
Group analysis of resting-state fMRI with dictionary learning: DictLearning
=====================================================

An example applying dictionary learning to resting-state data. This example
applies it to 20 subjects of the ADHD200 datasets.

Dictionary learning is a sparsity based decomposition method
for extracting spatial maps.

    * Gael Varoquaux et al.
    Multi-subject dictionary learning to segment an atlas of brain
    spontaneous activity
    Information Processing in Medical Imaging, 2011, pp. 562-573,
    Lecture Notes in Computer Science

Pre-prints for paper is available on hal
https://hal.inria.fr/inria-00588898/en/
"""
import numpy as np

### Load ADHD rest dataset ####################################################
from nilearn import datasets
# For linear assignment (should be moved in non user space...)

hcp_dataset = datasets.fetch_hcp_rest(data_dir='/volatile3',
                                      n_subjects=2)
# hcp_dataset = datasets.fetch_adhd(n_subjects=40)
mask = hcp_dataset.mask
func_filenames = hcp_dataset.func  # list of 4D nifti files for each subject

from sandbox.utils import output
# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      hcp_dataset.func[0])  # 4D data

### Apply DictLearning ########################################################
from nilearn.decomposition import DictLearning

n_components = 30

dict_learning = DictLearning(n_components=n_components, alpha=0.0,
                             l1_gamma=0.3,
                             smoothing_fwhm=4.,
                             mask=mask,
                             memory="nilearn_cache",
                             memory_level=3,
                             n_iter=1920,
                             shuffle=True,
                             verbose=10, random_state=0, n_jobs=1,
                             reduction=None)

estimator = dict_learning

components_imgs = []

print('[Example] Learning maps using %s model'
      % type(estimator).__name__)
estimator.fit(func_filenames[:40])
print('[Example] Dumping results')
components_img = estimator.masker_.inverse_transform(estimator.components_)
components_img.to_filename(output('dict_learning_resting_state.nii.gz'))
np.save(output('score'), estimator.score_)
# Debug info
(residual, sparsity, values) = estimator.debug_info_
np.save(output('residual'), residual)
np.save(output('sparsity'), sparsity)
np.save(output('values'), values)
### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, find_xyz_cut_coords
from nilearn.image import index_img

print('[Example] Displaying')


fig = plt.figure()
cut_coords = find_xyz_cut_coords(index_img(components_img, 6))
plot_prob_atlas(components_img, title="%s" % estimator.__class__.__name__,
                figure=fig, view_type='continuous',
                cut_coords=cut_coords, colorbar=False)

plt.show()