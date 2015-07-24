"""
PCA dimension reduction on single subjects
"""
import itertools

import numpy as np
from scipy import linalg
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.utils.extmath import randomized_svd
from sklearn.base import BaseEstimator

from ..input_data.base_masker import filter_and_mask
from ..input_data import MultiNiftiMasker, NiftiMasker, NiftiMapsMasker
from ..input_data.masker_validation import check_embedded_nifti_masker
from .._utils.cache_mixin import CacheMixin, cache
from .._utils.class_inspect import get_params
from .._utils.compat import izip
from .._utils.niimg_conversions import _iter_check_niimg


def make_pca_masker(masker, n_components=None, random_state=None):
    """Utility function to return a PCAMultiNiftiMasker from a MultiNiftiMasker

    Parameters
    ----------
    masker: MultiNiftiMasker or NiftiMasker,
        Masker to be copied from when creating new _PCAMultiNiftiMasker

    n_components: int
        In new masker, number of components to extract, for each 4D-Niimage.
        None means no reduction

    random_state: int or RandomState
        In new masker, pseudo number generator state used for random sampling.

    Returns
    ----------
    returned_masker: _PCAMultiNiftiMasker or MultiNiftiMasker,
        New masker with reduction capability
    """
    if not isinstance(masker, (NiftiMasker, MultiNiftiMasker)):
        raise ValueError("Expected type: *NiftiMasker but got type %s" %
                         type(masker))
    params = masker.get_params()
    if n_components is not None:
        returned_masker = PCAMultiNiftiMasker(n_components=n_components,
                                              random_state=random_state,
                                              **params)
    else:
        returned_masker = MultiNiftiMasker(**params)
    # If masker has already been fitted, intialize new masker with mask_img_
    if hasattr(masker, 'mask_img_'):
        returned_masker.mask_img = masker.mask_img_
        returned_masker.fit()
    return returned_masker


def session_pca(imgs, mask_img, parameters,
                n_components=20,
                confounds=None,
                memory_level=0,
                memory=Memory(cachedir=None),
                verbose=0,
                copy=True,
                sample_mask=None,
                random_state=0):
    """Filter, mask and compute PCA on Niimg-like objects

    This is an helper function whose first call `base_masker.filter_and_mask`
    and then apply a PCA to reduce the number of time series.

    Parameters
    ----------
    imgs: list of Niimg-like objects
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        List of subject data

    mask_img: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Mask to apply on the data

    parameters: dictionary
        Dictionary of parameters passed to `filter_and_mask`. Please see the
        documentation of the `NiftiMasker` for more informations.

    confounds: CSV file path or 2D matrix
        This parameter is passed to signal.clean. Please see the
        corresponding documentation for details.

    n_components: integer, optional
        Number of components to be extracted by the PCA

    memory_level: integer, optional
        Integer indicating the level of memorization. The higher, the more
        function calls are cached.

    memory: joblib.Memory
        Used to cache the function calls.

    verbose: integer, optional
        Indicate the level of verbosity (0 means no messages).

    copy: boolean, optional
        Whether or not data should be copied.

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    random_state:
    """

    data, affine = cache(
        filter_and_mask, memory,
        func_memory_level=3, memory_level=memory_level,
        ignore=['verbose', 'memory', 'memory_level', 'copy'])(
        imgs, mask_img, parameters,
        memory_level=memory_level,
        memory=memory,
        verbose=verbose,
        confounds=confounds,
        sample_mask=sample_mask,
        copy=copy)
    # If we project on a relatively small space, randomized_svd is faster
    if n_components <= data.shape[0] // 4:
        U, S, _ = cache(randomized_svd, memory, memory_level=memory_level,
                        func_memory_level=3)(
            data.T, n_components, random_state=random_state)
        U = U.T
    else:
        U, S, _ = cache(linalg.svd, memory, memory_level=memory_level,
                        func_memory_level=3)(
            data.T, full_matrices=False)
        U = U.T[:n_components].copy()
        S = S[:n_components]
    return U * S[:, np.newaxis], affine


class PCAMultiNiftiMasker(MultiNiftiMasker, CacheMixin):
    """Class for masking Niimg-like objects, with PCA compression in
     the direction of samples.
    This class is a monkey_patched version of MultiNiftiMasker,
    where filter_and_mask is replaced by session_pca

    Parameters
    ==========
    n_components: int
        Number of components to extract, for each 4D-Niimage.

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    mask_img: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Mask of the data. If not given, a mask is computed in the fit step.
        Optional parameters can be set using mask_args and mask_strategy to
        fine tune the mask extraction.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    mask_strategy: {'background' or 'epi'}, optional
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, and 'epi' if they
        are raw EPI images. Depending on this value, the mask will be
        computed from masking.compute_background_mask or
        masking.compute_epi_mask. Default is 'background'.

    mask_args : dict, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    Attributes
    ==========
    mask_img_: nibabel.Nifti1Image object
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    affine_: 4x4 numpy.ndarray
        Affine of the transformed image. If affine is different across
        subjects, contains the affine of the first subject on which other
        subject data have been resampled.

    See Also
    ========
    nilearn.image.resample_img: image resampling
    nilearn.masking.compute_epi_mask: mask computation
    nilearn.masking.apply_mask: mask application on image
    nilearn.signal.clean: confounds removal and general filtering of signals
    """

    def __init__(self, n_components=20,
                 random_state=None,
                 mask_img=None, smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0
                 ):
        MultiNiftiMasker.__init__(self, mask_img=mask_img,
                                  smoothing_fwhm=smoothing_fwhm,
                                  standardize=standardize,
                                  detrend=detrend,
                                  low_pass=low_pass,
                                  high_pass=high_pass, t_r=t_r,
                                  target_affine=target_affine,
                                  target_shape=target_shape,
                                  mask_strategy=mask_strategy,
                                  mask_args=mask_args,
                                  memory=memory, memory_level=memory_level,
                                  n_jobs=n_jobs, verbose=verbose
                                  )

        self.n_components = n_components
        self.random_state = random_state

    def transform_single_imgs(self, imgs, confounds=None, copy=True,
                              sample_mask=None):
        if not hasattr(self, 'mask_img_'):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)
        params = get_params(self.__class__, self)
        # Remove the mask-computing params: they are not useful and will
        # just invalid the cache for no good reason
        for name in ('mask_img', 'mask_args'):
            params.pop(name, None)

        data, _ = self._cache(session_pca, func_memory_level=2,
                              ignore=['verbose', 'memory', 'copy',
                                      'random_state'])(
            imgs, self.mask_img_,
            params,
            memory_level=self.memory_level,
            memory=self.memory,
            verbose=self.verbose,
            confounds=confounds,
            copy=copy,
            sample_mask=sample_mask,
            n_components=self.n_components,
            random_state=self.random_state
        )
        return data

    def transform_imgs(self, imgs_list, confounds=None, copy=True, n_jobs=1):
        """Prepare multi subject data in parallel

        Parameters
        ----------

        imgs_list: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            List of imgs file to prepare. One item per subject.

        confounds: list of confounds, optional
            List of confounds (2D arrays or filenames pointing to CSV
            files). Must be of same length than imgs_list.

        copy: boolean, optional
            If True, guarantees that output array has no memory in common with
            input array.

        n_jobs: integer, optional
            The number of cpus to use to do the computation. -1 means
            'all cpus'.
        """
        if not hasattr(self, 'mask_img_'):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)
        params = get_params(MultiNiftiMasker, self)
        # Remove the mask-computing params: they are not useful and will

        # just invalid the cache for no good reason
        for name in ('mask_img', 'mask_args'):
            params.pop(name, None)

        target_fov = None
        if self.target_affine is None:
            # Force resampling on first image
            target_fov = 'first'

        niimg_iter = _iter_check_niimg(imgs_list, ensure_ndim=None,
                                       atleast_4d=False,
                                       target_fov=target_fov,
                                       memory=self.memory,
                                       memory_level=self.memory_level,
                                       verbose=self.verbose)

        func = self._cache(session_pca, func_memory_level=2,
                           ignore=['verbose', 'memory', 'copy',
                                   'random_state'])
        if confounds is None:
            confounds = itertools.repeat(None, len(imgs_list))
        data = Parallel(n_jobs=n_jobs)(delayed(func)(
            imgs, self.mask_img_,
            parameters=params,
            memory_level=self.memory_level,
            memory=self.memory,
            verbose=self.verbose,
            confounds=confounds,
            copy=copy,
            n_components=self.n_components,
            random_state=self.random_state)
            for imgs, confounds in izip(niimg_iter, confounds))
        return list(zip(*data))[0]


class DecompositionEstimator(BaseEstimator):
    """Base class for decomposition estimator. Handles mask logic, provides
     transform and inverse_transform methods
    # XXX: Will provide score + sorting method in the future

    Parameters
    ==========
    n_components: int
        Number of components to extract, for each 4D-Niimage

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is put to 0 and their variance to 1 in the time dimension.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    low_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    mask_strategy: {'background' or 'epi'}, optional
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, and 'epi' if they
        are raw EPI images. Depending on this value, the mask will be
        computed from masking.compute_background_mask or
        masking.compute_epi_mask. Default is 'background'.

    mask_args : dict, optional
        If mask is None, these are additional parameters passed to
        masking.compute_background_mask or masking.compute_epi_mask
        to fine-tune mask computation. Please see the related documentation
        for details.

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    Attributes
    ==========
    `_pca_masker_`: instance of MultiNiftiMasker
        Masker used to filter and mask data as first step. If an instance of
        MultiNiftiMasker is given in `mask` parameter,
        this is a copy of it. Otherwise, a masker is created using the value
        of `mask` and other NiftiMasker related parameters as initialization.

    `mask_img_`: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.
    """

    def __init__(self, n_components=20,
                 random_state=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0):
        self.n_components = n_components
        self.random_state = random_state

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

    def fit(self, imgs, y=None):
        """Base fit for decomposition estimators : compute the embedded masker

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which the mask is calculated. If this is a list,
            the affine is considered the same for all.
        """
        if imgs is None or (hasattr(imgs, '__iter__') and len(imgs) == 0):
            # Common error that arises from a null glob. Capture
            # it early and raise a helpful message
            raise ValueError('Need one or more Niimg-like objects as input, '
                             'an empty list was given.')

        self.masker_ = check_embedded_nifti_masker(self)

        # Avoid warning with imgs != None
        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit(imgs)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_

    def _check_components_(self):
        if not hasattr(self, 'components_'):
            if self.__class__.__name__ == 'DecompositionEstimator':
                raise ValueError("Object has no components_ attribute. "
                                 "This may be because "
                                 "DecompositionEstimator is direclty "
                                 "being used.")
            else:
                raise ValueError("Object has no components_ attribute. "
                                 "This is probably because fit has not "
                                 "been called.")

    def transform(self, imgs, confounds=None):
        """Project the data into a reduced representation

        Parameters
        ----------
        imgs: iterable of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data to be projected

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details

        Returns
        ----------
        loadings: list of 2D ndarray,
            For each subject, each sample, loadings for each decomposition
            components
            shape: number of subjects * (number of scans, number of regions)
        """

        self._check_components_()
        components_img_ = self.masker_.inverse_transform(self.components_)
        nifti_maps_masker = NiftiMapsMasker(
            components_img_, self.masker_.mask_img_,
            resampling_target='maps')
        nifti_maps_masker.fit()
        # XXX: dealing properly with 4D/ list of 4D data?
        if confounds is None:
            confounds = [None] * len(imgs)
        return [nifti_maps_masker.transform(img, confounds=confound)
                for img, confound in zip(imgs, confounds)]

    def inverse_transform(self, loadings):
        """Use provided loadings to compute corresponding linear component
         combination in whole-brain voxel space

        Parameters
        ----------
        loadings: list of numpy array (n_samples x n_components)
            Component signals to tranform back into voxel signals

        Returns
        ----------
        reconstructed_imgs: list of nibabel.Nifti1Image
           For each loading, reconstructed Nifti1Image
        """
        if not hasattr(self, 'components_'):
            ValueError('Object has no components_ attribute. This is either '
                       'because fit has not been called or because'
                       '_DecompositionEstimator has direcly been used')
        self._check_components_()
        components_img_ = self.masker_.inverse_transform(self.components_)
        nifti_maps_masker = NiftiMapsMasker(
            components_img_, self.masker_.mask_img_,
            resampling_target='maps')
        nifti_maps_masker.fit()
        # XXX: dealing properly with 2D/ list of 2D data?
        return [nifti_maps_masker.inverse_transform(signal)
                for signal in loadings]
