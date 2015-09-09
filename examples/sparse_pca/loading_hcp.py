from os.path import join, expanduser

from nilearn.decomposition.base import mask_and_reduce
from nilearn.input_data import MultiNiftiMasker
from nilearn_sandbox import datasets as datasets_sandbox

imgs = datasets_sandbox.fetch_hcp_rest(n_subjects=1,
                                  data_dir=expanduser('~/data')).func
mask = join(expanduser('~/data'), 'HCP_mask', 'mask_img.nii.gz')

mask = MultiNiftiMasker(smoothing_fwhm=6, mask_img=mask).fit()

data = mask.transform(imgs)