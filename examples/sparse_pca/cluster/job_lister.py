import json
import os
from os.path import exists
from os.path import join, expanduser

from nilearn_sandbox import datasets as datasets_sandbox
from sklearn.utils import gen_batches

from nilearn import datasets
from nilearn._utils import check_niimg


def check_dataset(system_params, dataset, n_subjects):
    data_dir = system_params['data_dir']
    if dataset == 'adhd':
        dataset = datasets.fetch_adhd(data_dir=data_dir,
                                      n_subjects=n_subjects).func
        mask = join(data_dir, 'ADHD_mask', 'mask_img.nii.gz')
    elif dataset == 'hcp':
        dataset = datasets_sandbox.fetch_hcp_rest(data_dir=data_dir,
                                                  n_subjects=n_subjects).func
        mask = join(data_dir, 'HCP_mask', 'mask_img.nii.gz')
    elif dataset == 'hcp_reduced':
        dataset = datasets_sandbox.fetch_hcp_reduced(
                data_dir=data_dir, n_subjects=n_subjects).func
        mask = join(data_dir, 'HCP_mask', 'mask_img.nii.gz')
    elif dataset == 'adni':
        dataset = datasets_sandbox.fetch_adni_longitudinal_rs_fmri_DARTEL().func
        dataset = dataset[:n_subjects]
        mask = datasets_sandbox.fetch_adni_masks().fmri
    else:
        raise NotImplementedError
    return dataset, mask


def check_init(system_params, init=None):
    data_dir = system_params['data_dir']
    smith = datasets.fetch_atlas_smith_2009(data_dir=data_dir)
    if init == 'rsn70':
        init = smith.rsn70
        n_components = 70
    elif init == 'rsn20':
        init = smith.rsn20
        n_components = 20
    elif exists(init):
        init = init
        n_components = check_niimg(init).get_shape()[3]
    elif isinstance(init, int):
        init = None
        n_components = init
    else:
        raise ValueError('Init not supported')
    return init, n_components


def check_system_params(cachedir=expanduser('~/nilearn_cache'),
                        data_dir=expanduser('~/data'),
                        output_dir=expanduser('~/output/cluster_spca')):
    sparams = dict(cachedir=cachedir,
                   data_dir=data_dir,
                   output_dir=output_dir)
    sparams['output_dir'] = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for this_dir in (cachedir, data_dir):
        if not os.path.exists(this_dir):
            raise ValueError('Directory %s does not exists' % this_dir)
    return sparams


def _check_global_params(sparams,
                         dict_init='rsn70',
                         dataset='hcp',
                         smoothing_fwhm=4,
                         n_subjects=40):
    gparams = dict(dict_init=dict_init, dataset=dataset, n_subjects=n_subjects)

    dict_init, n_components = check_init(sparams, dict_init)

    dataset, mask = check_dataset(sparams, gparams['dataset'],
                                  gparams['n_subjects'])

    gparams['dict_init'] = dict_init
    gparams['n_components'] = n_components
    gparams['n_records'] = len(dataset)
    gparams['dataset'] = dataset
    gparams['mask'] = mask
    gparams['smoothing_fwhm'] = smoothing_fwhm

    return gparams


def _yield_exp_params(sparams, gparams,
                      alpha_list=[1],
                      feature_ratio_list=[1],
                      n_runs=1, n_slices=1):
    slices = gen_batches(gparams['n_records'],
                         gparams['n_records'] // n_slices)

    for random_state in range(n_runs):
        for this_slice in slices:
            for feature_ratio in feature_ratio_list:
                for alpha in alpha_list:
                    eparams = dict(random_state=random_state,
                                   slice=this_slice.indices(
                                           gparams['n_records']),
                                   alpha=alpha,
                                   feature_ratio=feature_ratio)
                    yield eparams


def _yield_warmup_params(sparams, gparams, n_slices=1):
    slices = gen_batches(gparams['n_records'],
                         gparams['n_records'] // n_slices)

    for this_slice in slices:
        eparams = dict(slice=this_slice.indices(gparams['n_records']))
        yield eparams


def json_dump(job_dir, dictionary, target):
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    with open(join(job_dir, target), 'w+') as f:
        json.dump(dictionary, f)


def build_json_job_list(job_dir,
                        cachedir='~/nilearn_cache',
                        data_dir='~/data',
                        output_dir='~/output/cluster_spca',
                        dict_init='rsn70',
                        dataset='hcp',
                        n_subjects=40,
                        alpha_list=[1.],
                        feature_ratio_list=[1.],
                        n_runs=1, n_slices=1,
                        warmup_slices=1,
                        ):
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    sparams = check_system_params(cachedir=cachedir,
                                  data_dir=data_dir,
                                  output_dir=output_dir)
    json_dump(job_dir, sparams, 'system.json')
    gparams = _check_global_params(sparams,
                                   dict_init=dict_init,
                                   dataset=dataset,
                                   n_subjects=n_subjects)
    json_dump(job_dir, gparams, 'global.json')
    for i, eparams in enumerate(_yield_exp_params(sparams, gparams,
                                                  alpha_list=alpha_list,
                                                  feature_ratio_list=
                                                  feature_ratio_list,
                                                  n_runs=n_runs,
                                                  n_slices=n_slices)):
        json_dump(job_dir, eparams, 'exp_%i.json' % i)

    for i, eparams in enumerate(_yield_warmup_params(sparams, gparams,
                                                     n_slices=warmup_slices)):
        json_dump(job_dir, eparams, 'warmup_%i.json' % i)

# if __name__ == '__main__':
#     timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#     build_json_job_list(join(expanduser('~/share/output/spca_cluster'), timestamp,
#                              'jobs'),
#                         alpha_list=np.logspace(-5, 0, 6),
#                         feature_ratio_list=np.linspace(1, 10, 5),
#                         warmup_slices=20)
