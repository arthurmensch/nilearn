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
import os
import numpy as np

direxp = 'work/output/dict_learning_partial_fit'
try:
    os.makedirs(direxp)
except:
    pass


### Load ADHD rest dataset ####################################################
from nilearn import datasets
# For linear assignment (should be moved in non user space...)

hcp_dataset = datasets.fetch_hcp_rest(data_dir='/volatile3',
                                      n_subjects=1)
mask = hcp_dataset.mask # '/storage/data/HCP_mask/mask_img.nii.gz'
func_filenames = hcp_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      hcp_dataset.func[0])  # 4D data

### Apply DictLearning ########################################################
from nilearn.decomposition import DictLearning
from os.path import join
n_components = 20

# alphas = [0.01, 0.1, 1.]
# l1_gammas = [3., 5., 10.]
#
# for alpha in alphas:
#     for l1_gamma in l1_gammas:
dict_learning = DictLearning(n_components=n_components, alpha=1,
                             smoothing_fwhm=4.,
                             mask=mask,
                             memory="nilearn_cache",
                             memory_level=3,
                             n_iter=20000,
                             verbose=10, random_state=0, n_jobs=4)

estimator = dict_learning

components_imgs = []

print('[Example] Learning maps using %s model'
      % type(estimator).__name__)
estimator.fit(func_filenames)
print('[Example] Dumping results')
dirsubexp = join(direxp, 'alpha_%f') % 6.
try:
    os.makedirs(dirsubexp)
except:
    pass
components_img = estimator.masker_\
    .inverse_transform(estimator.components_)
components_img.to_filename(join(dirsubexp,
                                'dict_learning_resting_state.nii.gz'))
# np.save(join(dirsubexp, 'score'), estimator.score_)
# Debug info
(residual, sparsity, values) = estimator.debug_info_
np.save(join(dirsubexp, 'residual'), residual)
np.save(join(dirsubexp, 'sparsity'), sparsity)
np.save(join(dirsubexp, 'values'), values)
### Visualize the results #####################################################
# Show some interesting components