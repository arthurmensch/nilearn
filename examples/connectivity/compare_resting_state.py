from os.path import join
import os
import time
from nilearn import datasets
from nilearn.decomposition import SparsePCA, DictLearning

output = '/volatile/arthur/output/compare'
try:
    os.makedirs(join(output, 'intermediary'))
except:
    pass

# dataset = datasets.fetch_hcp_rest(data_dir='/volatile3', n_subjects=1)
# mask = dataset.mask if hasattr(dataset, 'mask') else None
dataset = datasets.fetch_adhd(n_subjects=40)
smith = datasets.fetch_atlas_smith_2009()
dict_init = smith.rsn20
n_components = 20
func_filenames = dataset.func  # list of 4D nifti files for each subject

# from sandbox.utils import output
# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      dataset.func[0])  # 4D data


sparse_pca = SparsePCA(n_components=n_components, smoothing_fwhm=4.,
                       memory="nilearn_cache", dict_init=dict_init,
                       reduction_ratio=1.,
                       memory_level=3,
                       verbose=2,
                       random_state=0, l1_ratio=0.3,
                       n_epochs=1)

dict_learning = DictLearning(n_components=n_components, smoothing_fwhm=4.,
                             memory="nilearn_cache", dict_init=dict_init,
                             reduction_ratio='auto',
                             memory_level=3,
                             verbose=10,
                             random_state=0, alpha=6, max_nbytes=None,
                             n_epochs=1)

estimators = [sparse_pca]
components_imgs = []
timings = []
for estimator in estimators:
    print('[Example] Learning maps using %s model' % type(estimator).__name__)
    t0 = time.time()
    estimator.fit(func_filenames)
    timings.append(time.time() - t0)
    print('[Example] Dumping results')
    components_img = estimator.masker_.inverse_transform(estimator.components_)
    components_img.to_filename('%s_resting_state.nii.gz' %
                               type(estimator).__name__)
    components_imgs.append(components_img)

### Visualize the results #####################################################
# Show some interesting components
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
