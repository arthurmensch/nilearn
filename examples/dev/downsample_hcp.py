import os
from nilearn._utils import check_niimg
from nilearn.image import index_img
from nilearn.datasets.func import fetch_hcp_rest

data = fetch_hcp_rest(data_dir='/volatile3', n_subjects=40).func
reduction_ratio = 0.5

for i, data in enumerate(data[::4]):
    if i > 36:
        img = check_niimg(data)
        n_samples = img.get_shape()[3]
        img = index_img(img, slice(0, n_samples, 3))
        img.to_filename(os.path.expanduser('~/HCP_reduced/record_%i.nii.gz' % i))