"""
PCA dimension reduction on multiple subjects
"""
import itertools

import numpy as np
from scipy import linalg
from sklearn.externals.joblib import Memory
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_random_state
from sklearn.linear_model import LinearRegression

from ..input_data import NiftiMapsMasker
from ..input_data.base_masker import filter_and_mask
from .._utils.cache_mixin import CacheMixin, cache
from .._utils import as_ndarray
from .single_pca import SinglePCA


def session_pca(imgs, mask_img, parameters,
                n_components=20,
                confounds=None,
                memory_level=0,
                memory=Memory(cachedir=None),
                verbose=0,
                copy=True,
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
        Whether or not data should be copied

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    random_state:
    """

    data, affine = cache(
        filter_and_mask, memory,
        func_memory_level=2, memory_level=memory_level,
        ignore=['verbose', 'memory', 'memory_level', 'copy'])(
            imgs, mask_img, parameters,
            memory_level=memory_level,
            memory=memory,
            verbose=verbose,
            confounds=confounds,
            copy=copy)
    if n_components <= data.shape[0] // 4:
        U, S, _ = cache(randomized_svd, memory, memory_level=memory_level,
                        func_memory_level=2)(
            data.T, n_components, random_state=random_state)
    else:
        U, S, _ = cache(linalg.svd, memory, memory_level=memory_level,
                        func_memory_level=2)(
            data.T, full_matrices=False)
    U = U.T[:n_components].copy()
    S = S[:n_components]
    return U, S


class MultiPCA(SinglePCA, CacheMixin):
    """Perform Multi Subject Principal Component Analysis.

    Perform a PCA on each subject and stack the results. An optional Canonical
    Correlation Analysis can also be performed.

    Parameters
    ----------
    n_components: int
        Number of components to extract

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    mask: Niimg-like object, instance of NiftiMasker or MultiNiftiMasker, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    do_cca: boolean, optional
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

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
    ----------
    `masker_`: instance of MultiNiftiMasker
        Masker used to filter and mask data as first step. If an instance of
        MultiNiftiMasker is given in `mask` parameter,
        this is a copy of it. Otherwise, a masker is created using the value
        of `mask` and other NiftiMasker related parameters as initialization.

    `mask_img_`: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        The mask of the data. If no mask was given at masker creation, contains
        the automatically computed mask.

    `components_`: 2D numpy array (n_components x n-voxels)
        Array of masked extracted components. They can be unmasked thanks to
        the `masker_` attribute.
    """

    def __init__(self, n_components=20, smoothing_fwhm=None, mask=None,
                 do_cca=True, standardize=True, target_affine=None,
                 target_shape=None, low_pass=None, high_pass=None,
                 t_r=None, memory=Memory(cachedir=None), memory_level=0,
                 sorted=False,
                 n_jobs=1, verbose=0,
                 random_state=None
                 ):

        SinglePCA.__init__(self, n_components=n_components,
                           smoothing_fwhm=smoothing_fwhm,
                           mask=mask,
                           standardize=standardize, target_affine=target_affine,
                           target_shape=target_shape,
                           low_pass=low_pass, high_pass=high_pass,
                           t_r=t_r, memory=memory, memory_level=memory_level,
                           n_jobs=n_jobs, verbose=verbose,
                           random_state=random_state)
        self.do_cca = do_cca
        self.sorted = sorted

    def fit(self, imgs, y=None, confounds=None):
        """Compute the mask and the components

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which the PCA must be calculated. If this is a list,
            the affine is considered the same for all.
        """
        random_state = check_random_state(self.random_state)

        SinglePCA.fit(self, imgs, confounds=confounds)

        subject_pcas = self.components_list_
        subject_svd_vals = self.variance_list_

        if self.verbose:
            print("[MultiPCA] Learning group level PCA")
        if len(subject_pcas) > 1:
            if not self.do_cca:
                for subject_pca, subject_svd_val in \
                        zip(subject_pcas, subject_svd_vals):
                    subject_pca *= subject_svd_val[:, np.newaxis]
            data = np.empty((len(imgs) * self.n_components,
                            subject_pcas[0].shape[1]),
                            dtype=subject_pcas[0].dtype)
            for index, subject_pca in enumerate(subject_pcas):
                if self.n_components > subject_pca.shape[0]:
                    raise ValueError('You asked for %i components. '
                                     'This is larger than the single-subject '
                                     'data size (%d).' % (self.n_components,
                                                          subject_pca.shape[0]))
                data[index * self.n_components:
                     (index + 1) * self.n_components] = subject_pca
            data, variance, _ = self._cache(randomized_svd, func_memory_level=3)(
                data.T, n_components=self.n_components, random_state=random_state)
            # as_ndarray is to get rid of memmapping
            data = as_ndarray(data.T)
        else:
            data = subject_pcas[0]
            variance = subject_svd_vals[0]

        self.components_ = data
        self.variance_ = variance
        self.explained_variance_ratio_ = variance ** 2 / np.sum(self.variance_ ** 2)

        if self.sorted:
            self._sort_components(imgs, confounds)

        return self

    def transform(self, imgs, confounds=None):
        """ Project the data into a reduced representation

        Parameters
        ----------
        imgs: iterable of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data to be projected

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details
        """
        components_img_ = self.masker_.inverse_transform(self.components_)
        nifti_maps_masker = NiftiMapsMasker(
            components_img_, self.masker_.mask_img_,
            resampling_target='maps')
        nifti_maps_masker.fit()
        # XXX: dealing properly with 4D/ list of 4D data?
        if confounds is None:
            confounds = itertools.repeat(None, len(imgs))
        return [nifti_maps_masker.transform(img, confounds=confound)
                for img, confound in zip(imgs, confounds)]

    def inverse_transform(self, component_signals):
        """ Transform regions signals into voxel signals

        Parameters
        ----------
        component_signals: list of numpy array (n_samples x n_components)
            Component signals to tranform back into voxel signals
        """
        components_img_ = self.masker_.inverse_transform(self.components_)
        nifti_maps_masker = NiftiMapsMasker(
            components_img_, self.masker_.mask_img_,
            resampling_target='maps')
        nifti_maps_masker.fit()
        # XXX: dealing properly with 2D/ list of 2D data?
        return [nifti_maps_masker.inverse_transform(signal)
                for signal in component_signals]

    def _sort_components(self, imgs, confounds=None):
        """ Sort components by score obtained on test set imgs
        """
        score = self.score(imgs, confounds, per_component=True).mean(axis=0)
        self.score_ = -np.sort(-score)
        self.components_ = self.components_[np.argsort(-score)]

    def score(self, imgs, confounds=None, per_component=False):
        """Score function based on explained variance

        Parameters
        ----------
        imgs: iterable of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data to be scored

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details

        per_component: boolean,
            Specify whether the explained variance ratio is desired for each map or for the global set of components

        Returns
        -------
        score: ndarray or float,
            Holds the score for each subjects. Score is two dimensional if per_component is True. First dimension
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
            data = self._cache(
                filter_and_mask,
                func_memory_level=2,
                ignore=['verbose', 'copy'])(
                    img, self.mask_img_, self._get_filter_and_mask_parameters(),
                    memory_level=self.memory_level,
                    memory=self.memory,
                    verbose=self.verbose,
                    confounds=confound,
                    copy=True)[0]
            score[i] = self._score(data, per_component=per_component)
        return score if len(imgs) > 1 else score[0]

    def _score(self, data,
               per_component=False):
        """Score function based on explained variance

        Parameters
        ----------
        data: ndarray,
            Holds single subject data to be tested against components

        per_component: boolean,
            Specify whether the explained variance ratio is desired for each map or for the global set of components_

        Returns
        -------
        score: ndarray,
            Holds the score for each subjects. score is two dimensional if per_component = True
        """
        # If data is not standardized:
        data -= np.mean(data)

        full_var = np.sum(data ** 2)

        lr = LinearRegression(fit_intercept=False)
        if not per_component:
            residual_variance = lr.fit(self.components_.T, data.T).residues_.sum()
        else:
            # Per-component score : residues of projection onto each map
            residual_variance = np.zeros(self.n_components)
            for i in range(self.n_components):
                if np.any(self.components_[i]):
                    residual_variance[i] = lr.fit(self.components_.T[:, i][:, np.newaxis], data.T).residues_.sum()
                else:
                    # Setting score to 0 is component is empty
                    residual_variance[i] = full_var
        res = np.maximum(0., 1. - residual_variance / full_var)
        return res

