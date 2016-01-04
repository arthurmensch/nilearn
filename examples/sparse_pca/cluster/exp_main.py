import json
import sys
import os

from os.path import join, expanduser
from clusterlib.storage import sqlite3_dumps

from nilearn.decomposition import SparsePCA

SPARSEPCA = SparsePCA(batch_size=20,
                      n_epochs=10,
                      reduction_method=None,
                      reduction_ratio=1.,
                      verbose=10,
                      n_jobs=1)


def run(argv=None):
    if argv is None:
        argv = sys.argv
    job_dir = argv[1]
    job_number = argv[2]
    with open(join(job_dir, 'system.json'), 'r') as f:
        sparams = json.load(f)
    with open(join(job_dir, 'global.json'), 'r') as f:
        gparams = json.load(f)
    with open(join(job_dir, 'exp_%s.json' % job_number), 'r') as f:
        eparams = json.load(f)

    output_dir = sparams['output_dir']
    exp_folder = join(output_dir, 'exp_%s' % job_number)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    sys.stdout = open(join(exp_folder, 'output'), 'w')

    debug_folder = join(exp_folder, 'debug')

    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    spca = SparsePCA(batch_size=20,
                     n_epochs=10,
                     reduction_method=None,
                     reduction_ratio=1.,
                     verbose=10,
                     n_jobs=1)
    spca.set_params(
            memory=sparams['cachedir'],
            dict_init=gparams['dict_init'],
            mask=gparams['mask'],
            smoothing_fwhm=gparams['smoothing_fwhm'],
            alpha=eparams['alpha'],
            random_state=eparams['random_state'],
            feature_ratio=eparams['feature_ratio'],
            debug_folder=debug_folder,
    )

    imgs = gparams['dataset'][slice(*eparams['slice'])]
    with open(join(exp_folder, 'results'), 'w+') as f:
        f.write('%s' % spca)
        f.write('%s' % imgs)


if __name__ == "__main__":
    nosql_path = sys.argv[3]
    run()
    sqlite3_dumps({" ".join(sys.argv): "JOB DONE"}, nosql_path)
