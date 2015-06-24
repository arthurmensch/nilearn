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

### Load ADHD rest dataset ####################################################
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd()
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

### Apply DictLearning ########################################################
from nilearn.decomposition.dict_learning import DictLearning

n_components = 50
dict_learning = DictLearning(n_components=n_components, smoothing_fwhm=6.,
                             memory="nilearn_cache", memory_level=5,
                             threshold=1., verbose=10, random_state=0,
                             n_jobs=5, n_init=5, l1_ratio=0.1)
dict_learning.fit(func_filenames)

# Retrieve the independent components in brain space
components_img = dict_learning.masker_.inverse_transform(dict_learning.components_)
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
components_img.to_filename('dict_learning_resting_state.nii.gz')

### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import iter_img

for i, cur_img in enumerate(iter_img(components_img)):
    plot_stat_map(cur_img, display_mode="z", title="Map %d" % i, cut_coords=1,
                  colorbar=False)

plt.show()
