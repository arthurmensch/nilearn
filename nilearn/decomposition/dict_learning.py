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


class DictLearning(CanICA, CacheMixin):
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
                 l1_ratio=0.1,
                 batch_size=10,
                 # Common options
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 ):
        super(DictLearning, self).__init__(
            mask=mask, memory=memory, memory_level=memory_level,
            n_jobs=n_jobs, verbose=verbose, do_cca=do_cca,
            threshold=threshold, n_init=n_init,
            n_components=n_components, smoothing_fwhm=smoothing_fwhm,
            target_affine=target_affine, target_shape=target_shape)

        self.threshold = threshold
        self.random_state = random_state
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.standardize = standardize
        self.l1_ratio = l1_ratio
        self.batch_size = batch_size

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
        CanICA.fit(self, imgs, y=y, confounds=confounds)

        data = np.concatenate(self.data_flat_, axis=0)

        np.random.shuffle(data)

        n_samples = data.shape[0]

        self.components_ = cache(dict_learning_online, self.memory, memory_level=self.memory_level,
                                 ignore=['n_jobs'])(data, self.n_components,
                                                    n_iter=n_samples / self.batch_size,
                                                    return_code=False, dict_init=self.components_,
                                                    verbose=max(self.verbose - 1, 0),
                                                    shuffle=True, n_jobs=1,
                                                    l1_ratio=self.l1_ratio,
                                                    method="ols",
                                                    constraint="enet",
                                                    batch_size=self.batch_size,
                                                    random_state=self.random_state
                                                    )

        self.components_ = as_ndarray(self.components_)
        # flip signs in each component so that peak is +ve
        for component in self.components_:
            if component.max() < -component.min():
                component *= -1

        return self
