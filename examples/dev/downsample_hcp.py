import os
from nilearn._utils import check_niimg
from nilearn.image import index_img
from nilearn.datasets.func import fetch_hcp_rest

data = fetch_hcp_rest(data_dir=os.path.expanduser('~/data'), n_subjects=40).func

for i, data in enumerate(data[::2]):
    img = check_niimg(data)
    n_samples = img.get_shape()[3]
    img = index_img(img, slice(0, n_samples, 3))
    img.to_filename(os.path.expanduser('~/data/HCP_reduced/record_%i.nii.gz' % i))