"""
PCA dimension reduction on single subjects
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory

from .._utils.cache_mixin import CacheMixin


class SinglePCA(BaseEstimator, TransformerMixin, CacheMixin):

    def __init__(self, n_components=20, smoothing_fwhm=None, mask=None,
                 standardize=True, target_affine=None,
                 target_shape=None, low_pass=None, high_pass=None,
                 t_r=None, memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0,
                 random_state=None
                 ):
        self.mask = mask
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

        self.n_components = n_components
        self.smoothing_fwhm = smoothing_fwhm
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.standardize = standardize
        self.random_state = random_state

    def fit(selfself, imgs, y=None, confounds=None):
