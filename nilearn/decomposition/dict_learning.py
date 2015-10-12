"""
Dictionary learning estimator: Perform a map learning algorithm based on
component sparsity
"""

# Author: Arthur Mensch
# License: BSD 3 clause

from __future__ import division
import warnings

from math import ceil
from os.path import join
import time
from itertools import chain, repeat

import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.linear_model import Ridge

from sklearn.decomposition import dict_learning_online, sparse_encode

from sklearn.base import TransformerMixin
from sklearn.utils import gen_batches, check_random_state

from .canica import CanICA
from .._utils.cache_mixin import CacheMixin
from .base import DecompositionEstimator, mask_and_reduce, MaskReducer, \
    _close_and_remove


def _compute_loadings(components, data, in_memory=False):
    n_samples, n_features = data.shape
    n_components, n_features = components.shape
    ridge = Ridge(fit_intercept=None, alpha=0.)
    if in_memory:
        in_core_batch_size = n_samples
    else:
        in_core_batch_size = min(n_samples, n_samples / 10)
    batches = gen_batches(n_samples, in_core_batch_size)
    loadings = np.empty((n_components, n_samples), dtype='float64')
    for batch in batches:
        ridge.fit(components.T, np.asarray(data[batch].T))
        loadings[:, batch] = ridge.coef_.T

    S = np.sqrt(np.sum(loadings ** 2, axis=0))
    S[S == 0] = 1
    loadings /= S[np.newaxis, :]
    return loadings


class DictLearning(DecompositionEstimator, TransformerMixin, CacheMixin):
    """Perform a map learning algorithm based on component sparsity,
     over a CanICA initialization.  This yields more stable maps than CanICA.

    Parameters
    ----------
    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    n_components: int
        Number of components to extract

    n_epochs: float
        Number of epochs the algorithm should run on the data

    alpha: float, optional, default=1
        Sparsity controlling parameter

    dict_init: Niimg-like object, optional
        Initial estimation of dictionary maps. Would be computed from CanICA if
        not provided

    reduction_ratio: 'auto' or float, optional
        - Between 0. or 1. : controls compression of data, 1. means no
        compression
        - if set to 'auto', estimator will guess a good compression trade-off
        between speed and accuracy

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs: integer, optional, default=1
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    in_memory: boolean,
        Intermediary unmasked data will be
        stored as a tempory memory map

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    References
    ----------
    * Gael Varoquaux et al.
    Multi-subject dictionary learning to segment an atlas of brain spontaneous
    activity
    Information Processing in Medical Imaging, 2011, pp. 562-573,
    Lecture Notes in Computer Science

    """

    def __init__(self, n_components=20,
                 n_epochs=1, alpha=1, dict_init=None,
                 reduction_ratio='auto',
                 compression_type='svd',
                 # feature_compression=1,
                 power_iter=3,
                 forget_rate=1,
                 random_state=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, in_memory=True, temp_folder=None, verbose=0,
                 debug_folder=None,
                 batch_size=10,
                 parity=None,
                 ):
        DecompositionEstimator.__init__(self, n_components=n_components,
                                        random_state=random_state,
                                        mask=mask,
                                        # feature_compression=
                                        # feature_compression,
                                        smoothing_fwhm=smoothing_fwhm,
                                        standardize=standardize,
                                        detrend=detrend,
                                        low_pass=low_pass, high_pass=high_pass,
                                        t_r=t_r,
                                        target_affine=target_affine,
                                        target_shape=target_shape,
                                        mask_strategy=mask_strategy,
                                        mask_args=mask_args,
                                        memory=memory,
                                        memory_level=memory_level,
                                        n_jobs=n_jobs,
                                        in_memory=in_memory,
                                        verbose=verbose,
                                        parity=parity)
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.dict_init = dict_init
        self.reduction_ratio = reduction_ratio
        self.debug_folder = debug_folder
        self.batch_size = batch_size
        self.temp_folder = temp_folder
        self.compression_type = compression_type
        self.forget_rate = forget_rate
        self.power_iter = power_iter

    def _init_dict(self, imgs):
        if self.dict_init is None:
            canica = CanICA(n_components=self.n_components,
                            # CanICA specific parameters
                            do_cca=True, threshold=float(self.n_components),
                            n_init=1,
                            # mask parameter is not useful as we bypass masking
                            mask=self.masker_,
                            random_state=self.random_state,
                            memory=self.memory,
                            memory_level=self.memory_level,
                            n_jobs=self.n_jobs,
                            verbose=self.verbose
                            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # We use protected function _raw_fit as data
                # has already been unmasked
                canica.fit(imgs)
            components = canica.components_
        else:
            components = self.masker_.transform(self.dict_init)
        S = (components ** 2).sum(axis=1)
        S[S == 0] = 1
        components /= S[:, np.newaxis]
        self.components_init_ = components

    def _init_loadings(self, data):
        if not hasattr(self, 'components_init_'):
            raise ValueError('components_init_ need to be set before calling'
                             '_init_loadings')
        if self.debug_folder is not None:
            self.masker_.inverse_transform(self.components_init_).to_filename(
                join(self.debug_folder, 'init.nii.gz'))
        self._loadings_init = self._cache(_compute_loadings,
                                      func_memory_level=2)(self.components_init_,
                                                           data,
                                                           in_memory=False)

    def fit(self, imgs, y=None, confounds=None):
        """Compute the mask and the ICA maps across subjects

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrixf
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details
        """
        # Base logic for decomposition estimators
        DecompositionEstimator.fit(self, imgs)

        self.time_ = np.zeros(2)

        if self.verbose:
            print('[DictLearning] Learning initial components')
        t0 = time.time()
        self._init_dict(imgs)
        self.time_ += time.time() - t0

        if self.verbose:
            print('[DictLearning] Loading data')
        mask_reducer = MaskReducer(self.masker_,
                             reduction_ratio=self.reduction_ratio,
                             n_components=self.n_components,
                             compression_type=self.compression_type,
                             power_iter=self.power_iter,
                             random_state=self.random_state,
                             memory_level=max(0, self.memory_level - 1),
                             temp_folder=self.temp_folder,
                             n_jobs=self.n_jobs,
                             memory=self.memory,
                             in_memory=self.in_memory,
                             parity=self.parity)
        mask_reducer.fit(imgs, confounds)
        self.time_ = mask_reducer.time_
        self._raw_fit(mask_reducer.data_)

        if hasattr(mask_reducer, 'file_'):
            filename = mask_reducer.filename_
            file = mask_reducer.file_
            mask_reducer = None
            _close_and_remove(file, filename)

    def _raw_fit(self, data):
        """Compute the mask and the maps across subjects, using raw_data. Can
        only be called directly is dict_init and mask_img, or
        components_init_ is provided

        Parameters
        ----------
        data: ndarray,
            Shape (n_samples, n_features)
        """
        if not hasattr(self, 'time_'):
            self.time_ = np.zeros(2)
        if not hasattr(self, 'components_init_'):
            if self.dict_init is not None:
                if not hasattr(self, 'masker_'):
                    DecompositionEstimator.fit(self, None)
                self.components_init_ = self.masker_.transform(self.dict_init)
            else:
                raise ValueError('Calling _raw_fit directly is not possible '
                                 'whithout providing dict_init and mask_img, '
                                 'or setting components_init_')

        if self.n_epochs < 0:
            self.n_epochs = 1

        n_samples, n_features = data.shape
        if self.in_memory:
            in_core_batch_size = n_features
        else:
            in_core_batch_size = min(n_features, self.batch_size * 100)

        batches = gen_batches(n_features, in_core_batch_size)
        batches = chain.from_iterable(repeat(tuple(batches), self.n_epochs))
        inner_stats = None
        iter_offset = 0

        if self.verbose:
            print('[DictLearning] Computing initial loadings')
        t0 = time.time()
        self._init_loadings(data)
        self.time_[0] += time.time() - t0

        dict_init = self._loadings_init
        random_state = check_random_state(self.random_state)
        stream_range = random_state.permutation(n_features)

        if self.verbose:
            print('[DictLearning] Learning dictionary')
        for batch in batches:
            n_iter = (batch.stop - batch.start) // self.batch_size
            res = self._cache(dict_learning_online,
                              func_memory_level=2)(
                np.asarray(data[:, stream_range[batch]].T, order='C'),
                self.n_components,
                update_scheme='mean',
                forget_rate=self.forget_rate,
                alpha=self.alpha,
                n_iter=n_iter,
                batch_size=self.batch_size,
                method='cd',
                return_code=False,
                dict_init=dict_init,
                return_debug_info=self.debug_folder is not None,
                verbose=max(0, self.verbose - 1),
                random_state=self.random_state,
                return_inner_stats=True,
                inner_stats=inner_stats,
                iter_offset=iter_offset,
                shuffle=False,
                n_jobs=1,
                tol=0)
            self.time_[0] += time.time() - t0
            if self.debug_folder is not None:
                (dictionary, inner_stats), debug_info = res
                if not hasattr(self, 'debug_info_'):
                    self.debug_info_ = debug_info
                else:
                    for key in self.debug_info_:
                        self.debug_info_[key] += debug_info[key]
                np.save(join(self.debug_folder, 'dictionary'), dictionary)
                np.save(join(self.debug_folder, 'residuals'),
                        self.debug_info_['residuals'])
                np.save(join(self.debug_folder, 'density'),
                        self.debug_info_['density'])
                np.save(join(self.debug_folder, 'values'),
                        self.debug_info_['values'])
                np.save(join(self.debug_folder, 'time'), self.time_)
            else:
                dictionary, inner_stats = res
            iter_offset += n_iter
            dict_init = dictionary

        batches = gen_batches(n_features, in_core_batch_size)
        self.components_ = np.empty((self.n_components, n_features),
                                    dtype='float64')
        for batch in batches:
            t0 = time.time()
            self.components_[:, batch] = self._cache(sparse_encode,
                                                     func_memory_level=2,
                                                     ignore=['n_jobs'])(
                np.asarray(data[:, batch].T, order='C'),
                dictionary, algorithm='lasso_cd',
                alpha=self.alpha, n_jobs=self.n_jobs, check_input=False).T
            self.time_[0] += time.time() - t0

        # Normalize components
        S = np.sqrt(np.sum(self.components_ ** 2, axis=1))
        S[S == 0] = 1
        self.components_ /= S[:, np.newaxis]

        # flip signs in each composant positive part is l1 larger
        #  than negative part
        for component in self.components_:
            if np.sum(component > 0) < np.sum(component < 0):
                component *= -1

        return self