"""
Base class for decomposition estimators, utilies for masking and reducing group
data
"""
from __future__ import division

import atexit
import os
from math import ceil
from tempfile import mkstemp, mkdtemp
import warnings

import numpy as np
from scipy import linalg
from sklearn.externals.joblib import Memory, Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, randomized_range_finder
from sklearn.base import BaseEstimator

from ..input_data import NiftiMapsMasker
from ..input_data.masker_validation import check_embedded_nifti_masker
from .._utils.cache_mixin import CacheMixin, cache
from .._utils.niimg_conversions import check_niimg_4d
from .._utils.niimg import _safe_get_data
from .._utils.file_io import _delete_folder


class mask_and_reduce(object):
    """Mask and reduce provided data with provided masker, using a PCA

    Uses a PCA (randomized for small reduction ratio) or a range finding matrix
    on time series to reduce data size in time. For multiple image,
    the concatenation of data is returned, either as an ndarray or a memorymap
    (useful for big datasets that do not fit in memory).

    Parameters
    ----------
    masker: NiftiMasker or MultiNiftiMasker
        Masker to use to mask provided data

    imgs: list of Niimg-like objects
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        List of subject data

    confounds: CSV file path or 2D matrix
        This parameter is passed to signal.clean. Please see the
        corresponding documentation for details.

    reduction_ratio: 'auto' or float
        How to reduce the data. If 'auto', reduce it to the provided number of
        components. If float between 0. and 1., data will be reduced to the
        provided ratio. Default: 'auto'

    n_components: integer, optional
        Number of components to be extracted by the PCA

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    memory_level: integer, optional
        Integer indicating the level of memorization. The higher, the more
        function calls are cached.

    memory: joblib.Memory
        Used to cache the function calls.

    memory_mode: {'auto', 'array', 'memorymap'}
        Whether to return a memorymap or and ndarray. Default: 'auto'

    Retuns
    ------
    data: ndarray or memorymap
        Concatenation of reduced data
    """

    def __init__(self, masker, imgs, confounds=None,
                 reduction_ratio='auto',
                 feature_compression=1,
                 compression_type=None,
                 n_components=None, random_state=None,
                 memory_level=0,
                 memory=Memory(cachedir=None),
                 mock=False,
                 in_memory=True,
                 temp_folder=None,
                 n_jobs=1, power_iter=3,
                 parity=None):
        self.masker = masker
        self.imgs = imgs
        self.confounds = confounds
        self.reduction_ratio = reduction_ratio
        self.compression_type = compression_type
        self.n_components = n_components
        self.random_state = random_state
        self.memory_level = memory_level
        self.memory = memory
        self.in_memory = in_memory
        self.mock = mock
        self.n_jobs = n_jobs
        self.power_iter = power_iter
        self.temp_folder = temp_folder
        self.feature_compression = feature_compression
        self.parity = parity

    def __enter__(self):
        return_mmap = not self.in_memory

        mock = bool(self.mock)

        if mock and self.memory is None:
            warnings.warn('Mock run is useless if memory is disabled.')

        if not hasattr(self.imgs, '__iter__'):
            imgs = [self.imgs]
        else:
            imgs = self.imgs

        if self.reduction_ratio == 'auto':
            if self.n_components is None:
                reduction_ratio = 1
            else:
                reduction_ratio = 'auto'
        else:
            reduction_ratio = float(self.reduction_ratio)
            if reduction_ratio is None or reduction_ratio >= 1:
                reduction_ratio = 1

        if self.confounds is None:
            confounds = [None] * len(imgs)
        else:
            confounds = self.confounds

        if self.compression_type is None:
            if reduction_ratio == 1:
                compression_type = 'none'
            else:
                compression_type = 'range_finder'
        elif self.compression_type not in ['svd', 'range_finder', 'subsample',
                                           'none']:
            raise ValueError("`compression_type` should be `svd`"
                             "`range_finder`, `subsample` or `none`, got %s."
                             % self.compression_type)
        else:
            compression_type = self.compression_type

        # Precomputing number of samples for preallocation
        subject_n_samples = np.zeros(len(imgs), dtype='int')
        for i, img in enumerate(imgs):
            this_n_samples = check_niimg_4d(img).shape[3]
            if self.parity is not None:
                this_n_samples //= 2
            if reduction_ratio == 'auto':
                subject_n_samples[i] = min(self.n_components,
                                           this_n_samples)
            else:
                subject_n_samples[i] = int(ceil(this_n_samples
                                                * reduction_ratio))
        subject_limits = np.zeros(subject_n_samples.shape[0] + 1,
                                  dtype='int')
        subject_limits[1:] = np.cumsum(subject_n_samples)
        n_voxels = np.sum(_safe_get_data(self.masker.mask_img_))
        if self.feature_compression != 1.:
            random_state = check_random_state(self.random_state)
            selection = random_state.permutation(
                n_voxels)[:int(n_voxels * self.feature_compression)]
            n_voxels = selection.shape[0]
        else:
            selection = None
        n_samples = subject_limits[-1]

        if not mock:
            if return_mmap or self.n_jobs > 1:
                if self.temp_folder is None:
                    warnings.warn('Using system temporary folder : '
                                  'it may be too small')
                    self.temp_folder_ = mkdtemp()
                else:
                    if not os.path.exists(self.temp_folder):
                        raise ValueError('Temporary directory does not exist :'
                                         ' please create %s before using it.'
                                         % self.temp_folder)
                    self.temp_folder_ = self.temp_folder

                # We initialize data in memory or on disk
                self.file_, filename = mkstemp(dir=self.temp_folder_)
                atexit.register(lambda: _delete_folder(self.temp_folder_,
                                                       warn=True))
                data = np.memmap(filename, dtype='float64',
                                 order='F', mode='w+',
                                 shape=(n_samples, n_voxels))
            else:
                data = np.empty((n_samples, n_voxels), order='F',
                                dtype='float64')
        else:
            data = None

        Parallel(n_jobs=self.n_jobs)(delayed(_load_single_subject)(
            self.masker, data, subject_limits, subject_n_samples,
            compression_type, reduction_ratio,
            selection,
            mock,
            img, confound,
            self.memory,
            self.memory_level,
            self.random_state,
            i, self.power_iter,
            self.parity,
        ) for i, (img, confound) in enumerate(zip(imgs, confounds)))

        if not mock and not return_mmap and self.n_jobs > 1:
            # We used a memory map for multiprocessing, restoring
            data = np.array(data)
        return data

    def __exit__(self, type, value, traceback):
        if hasattr(self, 'file_'):
            # We use low level IO as we cannot use a file context manager
            # within this context manager
            os.close(self.file_)
        if hasattr(self, 'temp_folder_'):
            _delete_folder(self.temp_folder_)


def _load_single_subject(masker, data, subject_limits, subject_n_samples,
                         compression_type, reduction_ratio,
                         selection,
                         mock,
                         img, confound,
                         memory,
                         memory_level,
                         random_state,
                         i, power_iter,
                         parity):
    """Utility function for multiprocessing from mask_and_reduce"""
    this_data = masker.transform(img, confound)
    if parity == 0:
        if this_data.shape[0] % 2 == 1:
            this_data = this_data[::2]
            this_data = this_data[:-1]
        else:
            this_data = this_data[::2]
    elif parity == 1:
        this_data = this_data[1::2]

    random_state = check_random_state(random_state)

    if selection is not None:
        this_data = this_data[:, selection]

    if compression_type == 'svd':
        if subject_n_samples[i] <= this_data.shape[0] // 4:
            U, S, _ = cache(randomized_svd, memory,
                            memory_level=memory_level,
                            func_memory_level=1)(this_data.T,
                                                 subject_n_samples[i],
                                                 n_iter=power_iter,
                                                 random_state=random_state)
            U = U.T
        else:
            U, S, _ = cache(linalg.svd, memory,
                            memory_level=memory_level,
                            func_memory_level=1)(this_data.T,
                                                 full_matrices=False)
            U = U.T[:subject_n_samples[i]].copy()
            S = S[:subject_n_samples[i]]
        U = U * S[:, np.newaxis]
    elif compression_type == 'range_finder':
        if reduction_ratio == 1.:
            random_state = check_random_state(random_state)
            U = this_data.copy()
            random_state.shuffle(U)
        else:
            Q = randomized_range_finder(this_data, subject_n_samples[i], power_iter,
                                        random_state=random_state)
            U = Q.T.dot(this_data)
    elif compression_type == 'subsample':
        if reduction_ratio == 1.:
            U = this_data
        else:
            indices = np.floor(np.linspace(0, this_data.shape[0]-1,
                               subject_n_samples[i])).astype('int')
            U = this_data[indices]
    else:  # compression type = 'none'
        U = this_data
    if not mock:
        data[subject_limits[i]:subject_limits[i+1], :] = U


class DecompositionEstimator(BaseEstimator, CacheMixin):
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

    in_memory: boolean,
        Intermediary unmasked data will be
        stored as a tempory memory map

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
                 # feature_compression=1,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, in_memory=True,
                 verbose=0,
                 parity=None):
        self.n_components = n_components
        self.random_state = random_state

        # self.feature_compression = feature_compression

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
        self.in_memory = in_memory
        self.verbose = verbose
        self.parity = parity

    def fit(self, imgs, y=None, confounds=None, preload=False, temp_dir=None):
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

        if preload:
            if self.verbose:
                print('[mask and reduce] Performing mock run')
            with mask_and_reduce(self.masker_, imgs, confounds,
                                 memory_level=self.memory_level,
                                 memory=self.memory, mock=True,
                                 temp_folder=temp_dir,
                                 n_jobs=self.n_jobs) as data:
                data = None

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

    def _score(self, data, per_component=True):
        """Score function based on explained variance

        Parameters
        ----------
        data: ndarray,
            Holds single subject data to be tested against components

        per_component: boolean,
            Specify whether the explained variance ratio is desired for each
            map or for the global set of components_

        Returns
        -------
        score: ndarray,
            Holds the score for each subjects. score is two dimensional if
            per_component = True
        """
        return self._cache(explained_variance,
                           func_memory_level=2)(data,
                                                self.components_,
                                                per_component=per_component)

    def _fit_score(self, data):
        """Score components based on explained variance over data

        Parameters
        ----------
        data: ndarray,
            Holds single subject data to be tested against components

        Returns
        -------
        score: ndarray,
            Holds the score for each maps
        """
        self.score_ = self._score(data, per_component=True)

    def _sort_by_score(self, data):
        """ Sort components by score obtained on test set imgs
        """
        if not hasattr(self.score_, '__iter__'):
            self._fit_score(data)
        argsort = np.argsort(self.score_)[::-1]
        self.components_ = self.components_[argsort]
        self.score_ = self.score_[argsort]

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
        return [nifti_maps_masker.inverse_transform(loading)
                for loading in loadings]

    def score(self, imgs, confounds=None, per_component=True):
        """Score function based on explained variance on imgs.

        Should only be used by DecompositionEstimator derived classes

        Parameters
        ----------
        imgs: iterable of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data to be scored

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details

        per_component: boolean,
            Specify whether the explained variance ratio is desired for each
             map or for the global set of components

        Returns
        -------
        score: ndarray or float,
            Holds the score for each subjects. Score is two dimensional
             if per_component is True. First dimension
            is squeezed if the number of subjects is one
        """
        if not isinstance(imgs, tuple) and not isinstance(imgs, list):
            imgs = [imgs]
        if confounds is None:
            confounds = [None] * len(imgs)
        if per_component:
            score = np.zeros((len(imgs), self.n_components))
        else:
            score = np.zeros(len(imgs))
        for i, (img, confound) in enumerate(zip(imgs, confounds)):
            data = self.masker_.transform(img, confound)
            score[i] = self._score(data, per_component=per_component)
        return score if len(imgs) > 1 else score[0]


def explained_variance(X, components, per_component=False):
        """Score function based on explained variance

        Parameters
        ----------
        data: ndarray,
            Holds single subject data to be tested against components

        per_component: boolean,
            Specify whether the explained variance ratio is desired for each
            map or for the global set of components_

        Returns
        -------
        score: ndarray,
            Holds the score for each subjects. score is two dimensional if
            per_component = True
        """
        X_ = X[::10].copy()
        full_var = np.var(X_)
        n_components = components.shape[0]
        if per_component:
            S = np.sqrt(np.sum(components ** 2, axis=1))
            S[S == 0] = 1
            components_ = components / S[:, np.newaxis]
            res_var = np.zeros(n_components)
            cXT = components_.dot(X_.T)
            for i in range(n_components):
                res = X_ - np.outer(cXT[i],
                                    components_[i])
                res_var[i] = np.var(res)
            return np.maximum(0., 1. - res_var / full_var)
        else:
            lr = LinearRegression(fit_intercept=True)
            res_var = lr.fit(components.T,
                             X_.T).residues_.sum()
            return np.maximum(0., 1. - res_var / full_var)
