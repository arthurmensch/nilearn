"""
Dictionary learning estimator: Perform a map learning algorithm based on
component sparsity
"""

# Author: Arthur Mensch
# License: BSD 3 clause
from __future__ import division

import itertools
import os
from math import ceil

import numpy as np
import pickle
from sklearn.externals.joblib import Memory
from sklearn.decomposition import dict_learning_online


from sklearn.base import TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_range_finder
import time

from .._utils import as_ndarray
from .canica import CanICA
from .._utils.cache_mixin import CacheMixin
from .base import DecompositionEstimator, mask_and_reduce


class SparsePCA(DecompositionEstimator, TransformerMixin, CacheMixin):
    """Perform a map learning algorithm based on component sparsity,
     over a CanICA initialization.  This yields more stable maps than CanICA.

    Parameters
    ----------
    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    data: array-like, shape = [[n_samples, n_features], ...]
        Training vector, where n_samples is the number of samples,
        n_features is the number of features. There is one vector per
        subject.

    n_components: int
        Number of components to extract

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    alpha: float, optional, default=1
        Sparsity controlling parameter

    random_state: int or RandomState
        Pseudo number generator state used for random sampling.

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
                 n_epochs=1, l1_ratio=0.1, dict_init=None,
                 random_state=None,
                 shuffle=False,
                 batch_size=10,
                 reduction_ratio=1.,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 ):
        DecompositionEstimator.__init__(self, n_components=n_components,
                                        random_state=random_state,
                                        mask=mask,
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
                                        n_jobs=n_jobs, verbose=verbose,
                                        )

        self.n_epochs = n_epochs
        self.l1_ratio = l1_ratio
        self.dict_init = dict_init
        self.batch_size = batch_size
        self.reduction_ratio = reduction_ratio
        self.shuffle = shuffle

    def _init_dict(self, imgs, confounds=None):
        if self.dict_init is not None:
            self.dict_init_ = self.masker_.transform(self.dict_init)
        else:
            canica = CanICA(n_components=self.n_components,
                            # CanICA specific parameters
                            do_cca=True, threshold=float(self.n_components),
                            n_init=1,
                            mask=self.masker_,
                            random_state=self.random_state,
                            memory=self.memory,
                            memory_level=self.memory_level,
                            n_jobs=self.n_jobs,
                            verbose=self.verbose
                            )
            canica.fit(imgs, confounds=confounds)
            self.dict_init_ = canica.components_

    def fit(self, imgs, y=None, confounds=None, intermediary_directory=None):
        """Compute the mask and the ICA maps across subjects

        Parameters
        ----------
        imgs: list of Niimg-like objects
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Data on which PCA must be calculated. If this is a list,
            the affine is considered the same for all.

        confounds: CSV file path or 2D matrix
            This parameter is passed to nilearn.signal.clean. Please see the
            related documentation for details
        """
        # Base logic for decomposition estimators
        DecompositionEstimator.fit(self, imgs)

        random_state = check_random_state(self.random_state)

        n_epochs = int(self.n_epochs)
        if self.n_epochs < 1:
            raise ValueError('Number of n_epochs should be at least one,'
                             ' got {r}'.format(self.n_epochs))

        if self.verbose:
            print('[DictLearning] Initializating dictionary')
        self._init_dict(imgs, confounds)

        if confounds is None:
            confounds = [None] * len(imgs)

        inner_stats = None
        iter_offset = 0
        imgs_confounds = zip(imgs, confounds)
        dict_init = self.dict_init_

        imgs_confounds_list = itertools.chain(*[random_state.permutation(
            imgs_confounds) for _ in range(self.n_epochs)])

        for record, (img, confound) in enumerate(imgs_confounds_list):
            with mask_and_reduce(self.masker_, img, confound,
                                 reduction_ratio=self.reduction_ratio,
                                 n_components=self.n_components,
                                 compression_type='range_finder',
                                 random_state=self.random_state,
                                 memory_level=self.memory_level,
                                 memory=self.memory,
                                 max_nbytes=None) as data:
                n_iter = (data.shape[0] - 1) // self.batch_size + 1
                if self.verbose:
                    print('[DictLearning] Learning dictionary')
                (self.components_, inner_stats), debug_info = \
                    self._cache(dict_learning_online, func_memory_level=2)(
                    data,
                    self.n_components,
                    alpha=0.,
                    l1_ratio=self.l1_ratio,
                    n_iter=n_iter,
                    slowing=0.001 if not record else 0.,
                    batch_size=self.batch_size,
                    method='ridge',
                    dict_init=dict_init,
                    return_code=False,
                    verbose=max(0, self.verbose - 1),
                    random_state=self.random_state,
                    return_debug_info=True,
                    return_inner_stats=True,
                    inner_stats=inner_stats,
                    iter_offset=iter_offset,
                    shuffle=self.shuffle,
                    n_jobs=1,
                    project_dict=not record,
                    tol=0.
                    )
            iter_offset += n_iter
            dict_init = self.components_
            if not hasattr(self, 'debug_info_'):
                self.debug_info_ = debug_info
            else:
                debug_info_list = []
                for i, time_serie in enumerate(debug_info):
                   debug_info_list.append(np.concatenate((self.debug_info_[i],
                                                         time_serie), axis=0))
                self.debug_info_ = tuple(debug_info_list)

            if intermediary_directory is not None:
                self.masker_.inverse_transform(self.components_)\
                    .to_filename(os.path.join(intermediary_directory,
                                 'components_%i.nii.gz' % record))


        self.components_ = as_ndarray(self.components_)
        # flip signs in each composant positive part is l1 larger
        #  than negative part
        for component in self.components_:
            if np.sum(component[component > 0]) <\
                    - np.sum(component[component <= 0]):
                component *= -1

        if self.verbose:
            print('[DictLearning] Learning score')
        self._sort_components(data)

        return self