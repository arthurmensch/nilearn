"""
Group analysis of resting-state fMRI with dictionary learning: DictLearning
=====================================================

An example applying dictionary learning to resting-state data. This example applies it
to 20 subjects of the ADHD200 datasets.

Dictionary learning is a sparsity based decomposition method for extracting spatial maps.

    * Gael Varoquaux et al.
    Multi-subject dictionary learning to segment an atlas of brain spontaneous activity
    Information Processing in Medical Imaging, 2011, pp. 562-573, Lecture Notes in Computer Science

Pre-prints for paper is available on hal
https://hal.inria.fr/inria-00588898/en/
"""
import time
from sklearn.externals.joblib import Memory

### Load ADHD rest dataset ####################################################
from nilearn import datasets
# For linear assignment (should be moved in non user space...)
t0 = time.time()
adhd_dataset = datasets.fetch_adhd(n_subjects=10)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

### Apply DictLearning ########################################################
from nilearn.decomposition import DictLearning, CanICA
from nilearn.decomposition.multi_pca import MultiPCA

n_components = 30

dict_learning = DictLearning(n_components=n_components, smoothing_fwhm=6.,
                             memory="nilearn_cache", memory_level=2,
                             verbose=1,
                             random_state=0, alpha=3, max_nbytes=0,
                             n_epochs=1)
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache",  memory_level=2,
                verbose=1,
                n_init=1, threshold=3.)
multi_pca = MultiPCA(n_components=n_components, smoothing_fwhm=6.,
                     memory="nilearn_cache",  memory_level=2,
                     verbose=1)

estimators = [multi_pca, canica, dict_learning]

components_imgs = []

for estimator in estimators:
    print('[Example] Learning maps using %s model' % type(estimator).__name__)
    estimator.fit(func_filenames)
    print('[Example] Dumping results')
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_img.to_filename('%s_resting_state.nii.gz' %
                               type(estimator).__name__)
    components_imgs.append(components_img)

### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas, find_xyz_cut_coords, \
    plot_stat_map
from nilearn.image import index_img

mem = Memory(cachedir='~/nilearn_cache')

print('[Example] Displaying')

fig, axes = plt.subplots(ncols=len(estimators))
for estimator, cur_img, ax in zip(estimators, components_imgs, axes):
    plot_stat_map(index_img(cur_img, 0),
                    title="%s" % estimator.__class__.__name__,
                    axes=ax, colorbar=False, display_mode='x', cut_coords=1)
print("Elapsed time : %3is" % (time.time() - t0))
plt.show()