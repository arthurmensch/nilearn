"""
DictLearning
"""

# Author: Arthur Mensch
# License: BSD 3 clause
from ..input_data.base_masker import filter_and_mask

import numpy as np

from sklearn.externals.joblib import Memory, delayed, Parallel

from sklearn.decomposition import dict_learning_online
from .._utils.class_inspect import get_params
from .._utils import as_ndarray

from ..input_data import MultiNiftiMasker

from .canica import CanICA
from .._utils.cache_mixin import cache, CacheMixin

from sklearn.linear_model import Ridge
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.utils import gen_batches
from sklearn.decomposition.pca import RandomizedPCA

class DictLearning(CanICA, MiniBatchDictionaryLearning, CacheMixin):
    """Perform Dictionary Learning analysis.

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

    do_cca: boolean, optional
        Indicate if a Canonical Correlation Analysis must be run after the
        PCA.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    threshold: None, 'auto' or float
        If None, no thresholding is applied. If 'auto',
        then we apply a thresholding that will keep the n_voxels,
        more intense voxels across all the maps, n_voxels being the number
        of voxels in a brain volume. A float value indicates the
        ratio of voxels to keep (2. means keeping 2 x n_voxels voxels).

    n_init: int, optional
        The number of times the fastICA algorithm is restarted

    reduction: boolean, optional
        # XXX: Experimental
        Specify whether to reduce the dictionary by PCA before learning cde

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

    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    References
    ----------
    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299

    * G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
      datasets", IEEE ISBI 2010, p. 1177
    """

    def __init__(self, mask=None, n_components=20,
                 smoothing_fwhm=6, do_cca=True,
                 threshold='auto', n_init=10,
                 standardize=True,
                 random_state=0,
                 target_affine=None, target_shape=None,
                 low_pass=None, high_pass=None, t_r=None,
                 alpha=1,
                 batch_size=10,
                 n_iter=1000,
                 reduction=False,
                 # Common options
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 ):
        CanICA.__init__(self,
                        mask=mask, memory=memory, memory_level=memory_level,
                        n_jobs=n_jobs, verbose=max(0, verbose - 1), do_cca=do_cca,
                        threshold=threshold, n_init=n_init,
                        n_components=n_components, smoothing_fwhm=smoothing_fwhm,
                        target_affine=target_affine, target_shape=target_shape,
                        random_state=random_state, high_pass=high_pass, low_pass=low_pass,
                        t_r=t_r,
                        keep_data_mem=True,
                        standardize=standardize)
        self.reduction = reduction
        MiniBatchDictionaryLearning.__init__(self, n_components=n_components, alpha=alpha,
                                             n_iter=n_iter, batch_size=batch_size,
                                             fit_algorithm='lars',
                                             transform_alpha=alpha,
                                             verbose=max(0, verbose - 1),
                                             random_state=random_state,
                                             shuffle=True,
                                             n_jobs=1)

    def _init_dict(self, imgs, y=None, confounds=None):
        CanICA.fit(self, imgs, y=y, confounds=confounds)
        self.data_flat_ = np.concatenate(self.data_flat_, axis=0)
        if self.verbose:
            print('[DictLearning] Learning time serie')
        ridge = Ridge(alpha=1e-6, fit_intercept=None)
        ridge.fit(self.components_.T, self.data_flat_.T)
        self.dict_init = ridge.coef_.T
        S = np.sqrt(np.sum(self.dict_init ** 2, axis=0))
        self.dict_init /= S[np.newaxis, :]

    def fit(self, imgs, y=None, confounds=None):
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
        self._init_dict(imgs, y, confounds)

        if self.verbose:
            print('[DictLearning] Learning dictionary')
        MiniBatchDictionaryLearning.fit(self, self.data_flat_.T)
        if self.reduction:
            if self.verbose:
                print('[DictLearning] Reducing the dictionary')
            pca = RandomizedPCA(n_components=self.n_components, iterated_power=0,
                                random_state=self.random_state)
            self.components_ = pca.fit_transform(self.components_)
            data_trans = pca.transform(self.data_flat_.T)
        else:
            data_trans = self.data_flat_.T
        if self.verbose:
            print('[DictLearning] Learning code')
        self.components_ = MiniBatchDictionaryLearning.transform(self, data_trans).T
        if self.verbose:
            print('Done')

        self.components_ = as_ndarray(self.components_)
        # flip signs in each component so that peak is +ve
        for component in self.components_:
            if np.sum(component[component > 0]) < - np.sum(component[component <= 0]):
                component *= -1

        return self