import nibabel
import numpy as np

from nose.tools import assert_true
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Memory

from nilearn.input_data.masker_validation import check_embedded_nifti_masker
from nilearn.input_data import MultiNiftiMasker, NiftiMasker


class OwningClass(BaseEstimator):

    def __init__(self, mask=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 dummy=None):
        self.mask = mask

        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.mask_args = mask_args
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.dummy = dummy


def test_check_embedded_nifti_masker():
    owner = OwningClass()
    masker = check_embedded_nifti_masker(owner)
    assert_true(type(masker) == MultiNiftiMasker)

    for mask, multi in ((MultiNiftiMasker(), True), (NiftiMasker(), False)):
        owner = OwningClass(mask=masker)
        masker = check_embedded_nifti_masker(owner, multi=multi)
        assert_true(type(masker) == mask.__class__)
        for param_key in masker.get_params():
            if param_key not in ['memory', 'memory_level', 'n_jobs',
                                 'verbose']:
                assert_true(getattr(masker, param_key) ==
                            getattr(mask, param_key))
            else:
                assert_true(getattr(masker, param_key) ==
                            getattr(owner, param_key))

    shape = (6, 8, 10, 5)
    affine = np.eye(4)
    mask = nibabel.Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)
    owner = OwningClass(mask=mask)
    masker = check_embedded_nifti_masker(owner)
    assert_true(masker.mask_img is mask)

    data = np.zeros((9, 9, 9))
    data[2:-2, 2:-2, 2:-2] = 10
    imgs = nibabel.Nifti1Image(data, np.eye(4))
    mask = MultiNiftiMasker()
    mask.fit([[imgs]])
    owner = OwningClass(mask=mask)
    masker = check_embedded_nifti_masker(owner)
    assert_true(masker.mask_img == mask.mask_img_)
