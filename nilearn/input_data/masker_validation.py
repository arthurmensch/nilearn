import warnings

from .._utils.class_inspect import get_params
from . import MultiNiftiMasker, NiftiMasker


def check_embedded_nifti_masker(estimator, multi=True):
    """Base function for using a masker within a BaseEstimator class :
    This creates a masker from instance parameters :
    - If instance contains a mask image in mask parameter,
    we use this image as new masker mask_img, forwarding instance parameters to
    new masker : smoothing_fwhm, standardize, detrend, low_pass= high_pass,
    t_r, target_affine, target_shape, mask_strategy, mask_args,
    - If instance contains a masker in mask parameter, we use a copy of
    this masker, overidding all instance masker related parameters.
    In all case, we forward system parameters of instance to new masker :
    memory, memory_level, vebose, n_jobs

    Parameters
    ==========
        instance: object, instance of BaseEstimator
            The object that gives us the values of the parameters

        multi: boolean, default=True,
            Indicates whether to return a MultiNiftiMasker or a NiftiMasker

    Returns
    ==========
        masker: MultiNiftiMasker, NiftiMasker
            New masker
    """
    masker_type = MultiNiftiMasker if multi else NiftiMasker
    parameters = get_params(masker_type, estimator)
    mask = getattr(estimator, 'mask', None)

    if isinstance(mask, (NiftiMasker, MultiNiftiMasker)):
        # Creating (Multi)NiftiMasker from mask
        params = get_params(masker_type, mask)
        if multi and hasattr(estimator, 'n_jobs'):
            params['n_jobs'] = estimator.n_jobs
        masker = masker_type(memory=estimator.memory,
                             memory_level=estimator.memory_level,
                             verbose=estimator.verbose,
                             **params
                             )
        # Raising warning if user has provided a masker and diverging
        # masker arguments
        for param_key in masker.get_params():
            if param_key in parameters and getattr(masker, param_key)\
                    != parameters[param_key]:
                warnings.warn("Provided mask overrides"
                              " default/provided parameter %s" % param_key)
        # Forwarding potential attribute
        if hasattr(mask, 'mask_img_'):
            # Allow free fit of returned mask
            masker.mask_img = mask.mask_img_
    else:
        # Creating (Multi)NiftiMasker
        # with parameters extracted from estimator
        if multi and hasattr(estimator, 'n_jobs'):
            parameters['n_jobs'] = estimator.n_jobs
        masker = masker_type(memory=estimator.memory,
                             memory_level=estimator.memory_level,
                             verbose=estimator.verbose,
                             mask_img=mask, **parameters)

    return masker