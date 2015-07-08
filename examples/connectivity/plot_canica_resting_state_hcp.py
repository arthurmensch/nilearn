"""
Group analysis of resting-state fMRI with ICA: CanICA
=====================================================

An example applying CanICA to resting-state data. This example applies it
to 40 subjects of the ADHD200 datasets.

CanICA is an ICA method for group-level analysis of fMRI data. Compared
to other strategies, it brings a well-controlled group model, as well as a
thresholding algorithm controlling for specificity and sensitivity with
an explicit model of the signal. The reference papers are:

    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299

    * G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
      datasets", IEEE ISBI 2010, p. 1177

Pre-prints for both papers are available on hal
(http://hal.archives-ouvertes.fr)
"""

### Load ADHD rest dataset ####################################################
from nilearn.datasets import fetch_adhd
data_dir = '/volatile3'

adhd_data = fetch_adhd(n_subjects=1)
func_filenames = adhd_data.func

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      func_filenames[0])  # 4D data

### Apply CanICA ##############################################################
from nilearn.decomposition.canica import CanICA
from nilearn.decomposition.multi_pca import MultiPCA

n_components = 10
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="/volatile3/cache", memory_level=5,
                keep_data_mem=True, n_init=1,
                threshold=float(n_components),
                verbose=10, n_jobs=1, random_state=0,
                )
canica.fit(func_filenames)

# Retrieve the independent components in brain space
components_img = canica.masker_.inverse_transform(canica.components_)
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
components_img.to_filename('canica_resting_state.nii.gz')

print("Score")
print(canica.score(func_filenames, per_component=False))
print(canica.score_training(per_component=False))

### Visualize the results #####################################################
# Show some interesting components
# import matplotlib.pyplot as plt
# from nilearn.plotting import plot_stat_map
# from nilearn.image import iter_img
#
# for i, cur_img in enumerate(iter_img(components_img)):
#     plot_stat_map(cur_img, display_mode="z", title="IC %d" % i, cut_coords=1,
#                   colorbar=False)
#
# plt.show()
