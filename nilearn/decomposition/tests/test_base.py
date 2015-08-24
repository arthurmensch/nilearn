import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_true
import nibabel

from nilearn._utils.testing import assert_raises_regex
from nilearn.input_data import MultiNiftiMasker
from nilearn.decomposition.base import DecompositionEstimator


# def test_multi_nifti_masker():
#     shape = (6, 8, 10, 5)
#     affine = np.eye(4)
#     rng = np.random.RandomState(0)
#
#     # Create a "multi-subject" dataset
#     data = []
#     for i in range(8):
#         this_data = rng.normal(size=shape)
#         # Create fake activation to get non empty mask
#         this_data[2:4, 2:4, 2:4, :] += 10
#         data.append(nibabel.Nifti1Image(this_data, affine))
#
#     mask_img = nibabel.Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)
#     pca_multi_nifti_masker = PCAMultiNiftiMasker(mask_img=mask_img,
#                                                  n_components=3)
#
#     # Test fit on multiple image
#     components = pca_multi_nifti_masker.fit_transform(data)
#     assert_true(isinstance(data, list))
#     assert_true(components[0].shape == (3, 6 * 8 * 10))
#
#     # Test on single image
#     components = pca_multi_nifti_masker.transform(data[0])
#     assert_true(isinstance(components, np.ndarray))
#     assert_true(components.shape == (3, 6 * 8 * 10))
#
#     cov = components.dot(components.T)
#     cov_diag = np.zeros((3, 3))
#     for i in range(3):
#         cov_diag[i, i] = cov[i, i]
#     assert_array_almost_equal(cov - cov_diag, 0)
#
#
# def test_make_pca_masker():
#     shape = (6, 8, 10, 5)
#     affine = np.eye(4)
#     mask = nibabel.Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)
#     masker = MultiNiftiMasker(mask_img=mask)
#     returned_masker = make_pca_masker(masker, n_components=3,
#                                       random_state=0)
#     # Type and argument checking
#     assert_true(isinstance(returned_masker, PCAMultiNiftiMasker))
#     for key in masker.get_params():
#         if key is not 'memory':
#             assert_true(getattr(returned_masker, key) == getattr(masker, key))
#     assert_true(returned_masker.n_components == 3)
#     assert_true(returned_masker.random_state == 0)
#     masker.fit()
#
#     # Test mask_img transmission
#     returned_masker = make_pca_masker(masker, n_components=3,
#                                       random_state=0)
#     assert_true(returned_masker.mask_img == masker.mask_img_)


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