from os.path import join
import os
import numpy as np
import pickle
from decomposition import DictLearning
from nilearn import datasets
from nilearn.decomposition import SparsePCA
# For linear assignment (should be moved in non user space...)

output = '/volatile/arthur/output/hcp_fast'
try:
    os.makedirs(join(output, 'intermediary'))
except:
    pass

# dataset = datasets.fetch_hcp_rest(data_dir='/volatile3', n_subjects=1)
# mask = dataset.mask if hasattr(dataset, 'mask') else None
dataset = datasets.fetch_adhd(n_subjects=40)
rsn70 = '/volatile/arthur/work/data/rsn70.nii.gz'
func_filenames = dataset.func  # list of 4D nifti files for each subject

# from sandbox.utils import output
# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      dataset.func[0])  # 4D data

n_components = 70

sparse_pca = SparsePCA(n_components=n_components,
                       l1_ratio=0.3,
                       smoothing_fwhm=4.,
                       reduction_ratio=1.,
                       batch_size=10,
                       n_epochs=1,
                       dict_init=rsn70,
                       memory="nilearn_cache",
                       memory_level=3,
                       shuffle=False,
                       verbose=2, random_state=0)

dict_learning = DictLearning(n_components=n_components, smoothing_fwhm=4.,
                             memory="nilearn_cache", dict_init=rsn70,
                             memory_level=3,
                             verbose=2,
                             random_state=0, alpha=10, max_nbytes=0,
                             n_epochs=2)

estimator = sparse_pca

print('[Example] Learning maps using %s model'
      % type(estimator).__name__)
estimator.fit(func_filenames,
              intermediary_directory=join(output, 'intermediary'))
print('[Example] Dumping results')
components_img = estimator.masker_.inverse_transform(estimator.components_)
components_img.to_filename(join(output, 'dict_learning_resting_state.nii.gz'))
np.save(join(output, 'score'), estimator.score_)
# Debug info
(residual, sparsity, values) = estimator.debug_info_
np.save(join(output, 'residual'), residual)
np.save(join(output, 'sparsity'), sparsity)
np.save(join(output, 'values'), values)
with open(join(output, 'parameters'), mode='w+') as f:
    pickle.dump(sparse_pca.get_params(), f)
with open(join(output, 'dataset'), mode='w+') as f:
    pickle.dump(dataset, f)
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