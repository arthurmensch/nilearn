import json
import os
from os.path import join

import sys

from clusterlib.storage import sqlite3_dumps

from nilearn.decomposition.base import BaseDecomposition, mask_and_reduce


def run(argv=None):
    if argv is None:
        argv = sys.argv
    job_dir = argv[1]
    job_number = argv[2]

    with open(join(job_dir, 'system.json'), 'r') as f:
        sparams = json.load(f)
    with open(join(job_dir, 'global.json'), 'r') as f:
        gparams = json.load(f)
    with open(join(job_dir, 'warmup_%s.json' % job_number), 'r') as f:
        eparams = json.load(f)

    output_dir = sparams['output_dir']
    exp_folder = join(output_dir, 'warmup_%s' % job_number)

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    sys.stdout = open(join(exp_folder, 'stdout'), 'w')
    sys.stderr = open(join(exp_folder, 'stderr'), 'w')


    debug_folder = join(exp_folder, 'debug')

    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    decomposition_estimator = BaseDecomposition(verbose=10,
                                                n_jobs=1,
                                                mask=gparams['mask'],
                                                smoothing_fwhm=gparams[
                                                    'smoothing_fwhm'])
    imgs = gparams['dataset'][slice(*eparams['slice'])]
    decomposition_estimator.fit(imgs)
    masker = decomposition_estimator.masker_

    _ = mask_and_reduce(masker, imgs,
                        reduction_method=None,
                        memory=sparams['cachedir'],
                        memory_level=2,
                        n_jobs=1)


if __name__ == "__main__":
    nosql_path = sys.argv[3]
    run()
    sqlite3_dumps({" ".join(sys.argv): "JOB DONE"}, nosql_path)

