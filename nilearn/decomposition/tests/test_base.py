import numpy as np
from nose.tools import assert_true
import nibabel

from nilearn._utils.testing import assert_raises_regex
from nilearn.input_data import MultiNiftiMasker
from nilearn.decomposition.base import DecompositionEstimator


def test_decomposition_estimator():
    shape = (6, 8, 10, 5)
    affine = np.eye(4)
    rng = np.random.RandomState(0)
    data = []
    for i in range(8):
        this_data = rng.normal(size=shape)
        # Create fake activation to get non empty mask
        this_data[2:4, 2:4, 2:4, :] += 10
        data.append(nibabel.Nifti1Image(this_data, affine))
    mask = nibabel.Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)
    masker = MultiNiftiMasker(mask_img=mask)
    decomposition_estimator = DecompositionEstimator(mask=masker,
                                                     n_components=3)
    decomposition_estimator.fit(data)
    assert_true(decomposition_estimator.mask_img_ == mask)
    assert_true(decomposition_estimator.mask_img_ ==
           decomposition_estimator.masker_.mask_img_)

    # Testing fit on data
    masker = MultiNiftiMasker()
    decomposition_estimator = DecompositionEstimator(mask=masker,
                                                     n_components=3)
    decomposition_estimator.fit(data)
    assert_true(decomposition_estimator.mask_img_ ==
           decomposition_estimator.masker_.mask_img_)

    assert_raises_regex(ValueError,
                        "Object has no components_ attribute. "
                        "This may be because "
                        "DecompositionEstimator is direclty "
                        "being used.",
                        decomposition_estimator.transform, data)
    assert_raises_regex(ValueError,
                        'Need one or more Niimg-like objects as input, '
                        'an empty list was given.',
                        decomposition_estimator.fit, [])