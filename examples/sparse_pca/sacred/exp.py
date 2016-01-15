import time

from nilearn.decomposition import SparsePCA
from nilearn.decomposition.base import mask_and_reduce


def main():
    spca = SparsePCA(batch_size=20,
                     n_epochs=5,
                     reduction_method=None,
                     reduction_ratio=1.,
                     memory_level=2,
                     warmup=False,
                     verbose=10,
                     n_jobs=1)
