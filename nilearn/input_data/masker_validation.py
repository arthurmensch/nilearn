import warnings

from .._utils.class_inspect import get_params
from . import MultiNiftiMasker, NiftiMasker


def check_embedded_nifti_masker(estimator, multi_subject=True):
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
    masker_type = MultiNiftiMasker if multi_subject else NiftiMasker
    estimator_params = get_params(masker_type, estimator)
    mask = getattr(estimator, 'mask', None)

    if isinstance(mask, (NiftiMasker, MultiNiftiMasker)):
        # Creating (Multi)NiftiMasker from provided masker
        masker_params = get_params(masker_type, mask)
        new_masker_params = masker_params
    else:
        # Creating (Multi)NiftiMasker
        # with parameters extracted from estimator
        new_masker_params = estimator_params
        new_masker_params['mask_img'] = mask
    # Forwarding system parameters of instance to new masker in all case
    if multi_subject and hasattr(estimator, 'n_jobs'):
        # For MultiNiftiMasker only
        new_masker_params['n_jobs'] = estimator.n_jobs
    new_masker_params['memory'] = estimator.memory
    new_masker_params['memory_level'] = estimator.memory_level
    new_masker_params['verbose'] = estimator.verbose

    # Raising warning if masker override parameters
    conflict_string = ""
    for param_key in estimator_params.keys():
        if new_masker_params[param_key] != estimator_params[param_key]:
            conflict_string += "Parameter {!r} :\n" \
                               "    Masker parameter {!r}" \
                               " - overriding estimator parameter {!r}\n" \
                .format(param_key, new_masker_params[param_key],
                        estimator_params[param_key])

    if conflict_string != "":
        warn_str = "Overriding provided-default estimator parameters with"\
                   " provided masker parameters :\n"\
                   + conflict_string
        warnings.warn(warn_str)
    masker = masker_type(**new_masker_params)

    # Forwarding potential attribute of provided masker
    if hasattr(mask, 'mask_img_'):
        # Allow free fit of returned mask
        masker.mask_img = mask.mask_img_

    return masker