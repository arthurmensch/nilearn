"""
Dictionary learning estimator: Perform a map learning algorithm based on
component sparsity
"""

# Author: Arthur Mensch
# License: BSD 3 clause
from __future__ import division
import itertools
import os
import time
from os.path import join
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.decomposition.dict_learning import MiniBatchDictionaryLearning
from sklearn.externals.joblib import Memory
from sklearn.utils import check_random_state
from .base import BaseDecomposition, mask_and_reduce
from .canica import CanICA
from .._utils.cache_mixin import CacheMixin


class SparsePCA(BaseDecomposition, TransformerMixin, CacheMixin):
    """Perform a map learning algorithm based on component sparsity,
     over a CanICA initialization.  This yields more stable maps than CanICA.

    Parameters
    ----------
    mask: Niimg-like object or MultiNiftiMasker instance, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters.

    reduction_method: 'svd' | 'rf' | 'ss' | None

    reduction_ratio: [0, 1] or 'auto'

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
                 n_epochs=1, dict_init=None,
                 alpha=0.,
                 update_scheme='mean',
                 reduction_method=None,
                 reduction_ratio=1.,
                 forget_rate=1,
                 random_state=None,
                 batch_size=10,
                 debug_folder=None,
                 mask=None, smoothing_fwhm=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0, feature_ratio=1,
                 ):
        BaseDecomposition.__init__(self, n_components=n_components,
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

        self.alpha = alpha
        self.update_scheme = update_scheme
        self.forget_rate = forget_rate
        self.n_epochs = n_epochs
        self.dict_init = dict_init
        self.batch_size = batch_size
        self.reduction_ratio = reduction_ratio
        self.reduction_method = reduction_method
        self.feature_ratio = feature_ratio
        self.debug_folder = debug_folder

    def _init_dict(self, imgs, confounds=None):
        if self.dict_init is not None:
            self._dict_init = self.masker_.transform(self.dict_init)
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
            self._dict_init = canica.components_
        S = np.sqrt((self._dict_init ** 2).sum(axis=1))
        S[S == 0] = 1
        self._dict_init /= S[:, np.newaxis]
        if self.debug_folder is not None:
            self.masker_.inverse_transform(self._dict_init).to_filename(join(
                self.debug_folder, 'init.nii.gz'))

    def fit(self, imgs, y=None, confounds=None, probe=None):
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
        BaseDecomposition.fit(self, imgs)

        random_state = check_random_state(self.random_state)

        n_epochs = int(self.n_epochs)
        if self.n_epochs < 1:
            raise ValueError('Number of n_epochs should be at least one,'
                             ' got {r}'.format(self.n_epochs))

        if confounds is None:
            confounds = [None] * len(imgs)

        if self.debug_folder is not None:
            os.mkdir(join(self.debug_folder, 'intermediary'))

        if self.verbose:
            print('[DictLearning] Initializating dictionary')
        self._init_dict(imgs, confounds)

        iter_offset = 0
        dict_init = self._dict_init

        self.time_ = np.zeros(2)

        incr_spca = MiniBatchDictionaryLearning(n_components=self.n_components,
                                                random_state=random_state,
                                                alpha=self.alpha,
                                                learning_rate=1,
                                                feature_ratio=
                                                self.feature_ratio,
                                                fit_algorithm='ridge',
                                                transform_algorithm='ridge',
                                                l1_ratio=1,
                                                batch_size=self.batch_size,
                                                dict_init=dict_init,
                                                shuffle=True,
                                                n_jobs=1,
                                                tol=0.,
                                                n_iter=10,
                                                debug_info=True,
                                                verbose=max(0,
                                                            self.verbose - 1))
        t0 = time.time()
        data_list = mask_and_reduce(self.masker_, imgs, confounds,
                                    reduction_ratio=self.reduction_ratio,
                                    n_components=self.n_components,
                                    reduction_method=self.reduction_method,
                                    random_state=self.random_state,
                                    memory=self.memory,
                                    memory_level=max(0, self.memory_level - 1),
                                    as_shelved_list=True,
                                    n_jobs=self.n_jobs)
        self.time_[1] += time.time() - t0

        data_list = itertools.chain(*[random_state.permutation(
            data_list) for _ in range(n_epochs)])

        for record, data in enumerate(data_list):
            t0 = time.time()
            data = data.get()
            self.time_[1] += time.time() - t0
            n_iter = (data.shape[0] - 1) // self.batch_size + 1
            if self.verbose:
                print('[DictLearning] Learning dictionary')
            t0 = time.time()
            incr_spca.partial_fit(data, deprecated=False)
            self.time_[0] += time.time() - t0
            self.components_ = incr_spca.components_

            if self.debug_folder is not None and record % 10 == 0:
                components_temp = self.components_.copy()
                for component in components_temp:
                    if np.sum(component > 0) < np.sum(component < 0):
                        component *= -1
                components_img = self.masker_.inverse_transform(
                    components_temp)
                components_img.to_filename(join(self.debug_folder,
                                                'intermediary',
                                                'at_%i.nii.gz' % record))
                np.save(join(self.debug_folder, 'residuals'),
                        np.array(incr_spca.debug_info_['residuals']))
                if probe is not None:
                    if not hasattr(self, 'score_'):
                        self.score_ = []
                    score = np.mean(self.score(probe))
                    self.score_.append(score)
                    np.save(join(self.debug_folder, 'score'))
            iter_offset += n_iter

        S = np.sqrt(np.sum(self.components_ ** 2, axis=1))
        S[S == 0] = 1
        self.components_ /= S[:, np.newaxis]
        # flip signs in each composant positive part is l1 larger
        #  than negative part
        for component in self.components_:
            if np.sum(component > 0) < np.sum(component < 0):
                component *= -1

        if self.verbose:
            print('[DictLearning] Learning score')
        score = self._raw_score(data)
        if self.debug_folder is not None:
            np.save(join(self.debug_folder, 'score'), score)

        return self
