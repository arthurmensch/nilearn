import nibabel
import numpy as np
from numpy.testing import assert_array_almost_equal, \
    assert_raises
from sklearn.utils.linear_assignment_ import linear_assignment
from nilearn.input_data import MultiNiftiMasker
from nilearn._utils import check_niimg_4d


def align_flat(reference, target_list, inplace=False):
    if not isinstance(target_list, list):
        return align_one_to_one(reference, target_list, inplace=inplace)
    if not inplace:
        res = []
    for i, target_components in enumerate(target_list):
        if inplace:
            align_one_to_one(reference, target_components,
                             inplace=True)
        else:
            res.append(align_one_to_one(reference, target_components,
                                        inplace=False))
    if inplace:
        res = target_list
    return res


def align_one_to_one(base_components, target_components, inplace=False):
    base_S = np.sqrt(np.sum(base_components ** 2, axis=1))
    base_S[base_S == 0] = 1
    base_components /= base_S[:, np.newaxis]

    target_S = np.sqrt(np.sum(target_components ** 2, axis=1))
    target_S[target_S == 0] = 1
    target_components /= target_S[:, np.newaxis]

    K = base_components.dot(target_components.T)
    indices = linear_assignment(-K)

    base_components *= base_S[:, np.newaxis]
    target_components *= target_S[:, np.newaxis]

    if inplace:
        target_components[indices[:, 0]] = target_components[indices[:, 1]]
    else:
        new_target = np.empty_like(target_components)
        new_target[indices[:, 0]] = target_components[indices[:, 1]]
        target_components = new_target
    return target_components


def align_nii(masker, reference_img, target_imgs):
    reference_flat = masker.transform(reference_img)
    target_flats = masker.transform(target_imgs)
    align_flat(reference_flat, target_flats, inplace=True)
    if isinstance(target_flats, list):
        return [masker.inverse_transform(target_flat) for target_flat
                in target_flats]
    else:
        return masker.inverse_transform(target_flats)


def align_with_last_nii(masker, imgs):
    res = align_nii(masker, imgs[-1], imgs[:-1])
    res.append(check_niimg_4d(imgs[-1]))
    return res


def test_align_nii():
    affine = np.eye(4)
    rng = np.random.RandomState(0)
    a = rng.randn(10, 5 * 5 * 5)
    b = rng.permutation(a)
    c = rng.permutation(a)
    masker = MultiNiftiMasker(mask_img=nibabel.Nifti1Image(np.ones((5, 5, 5)),
                                                           affine=affine))
    masker.fit()
    img_a = masker.inverse_transform(a)
    img_b = masker.inverse_transform(b)
    img_c = masker.inverse_transform(c)
    new_img_b = align_nii(masker, img_a, img_b)
    new_b = masker.transform(new_img_b)
    assert_array_almost_equal(a, new_b)
    results = align_with_last_nii(masker, (img_b, img_c, img_a))
    new_b = masker.transform(results[0])
    new_c = masker.transform(results[1])
    assert_array_almost_equal(a, new_b)
    assert_array_almost_equal(a, new_c)



def test_align_one_to_one():
    rng = np.random.RandomState(0)
    a = rng.rand(10, 100)
    a_copy = a.copy()
    b = rng.permutation(a)
    b_copy = b.copy()
    c = align_one_to_one(a, b, inplace=False)
    assert_array_almost_equal(a, c)
    assert_array_almost_equal(a, a_copy)
    assert_array_almost_equal(b, b_copy)
    align_one_to_one(a, b, inplace=True)
    assert_array_almost_equal(a, b)
    assert_array_almost_equal(a, a_copy)
    assert_raises(AssertionError, assert_array_almost_equal, b, b_copy)


def test_align_flat():
    rng = np.random.RandomState(0)
    ref = rng.rand(10, 100)
    b = rng.permutation(ref)
    c = rng.permutation(ref)
    target_list = [b, c]
    target_list_copy = [b.copy(), c.copy()]
    aligned_target_list = align_flat(ref, target_list, inplace=False)
    for target, target_copy in zip(target_list, target_list_copy):
        assert_array_almost_equal(target, target_copy)
    for target in aligned_target_list:
        assert_array_almost_equal(ref, target)

