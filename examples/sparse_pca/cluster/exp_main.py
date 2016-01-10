import json
import os
import sys
from os.path import join

from clusterlib.storage import sqlite3_dumps

from nilearn.decomposition import SparsePCA
from sklearn.model_selection import train_test_split

SPARSEPCA = SparsePCA(batch_size=20,
                      n_epochs=5,
                      reduction_method=None,
                      reduction_ratio=1.,
                      memory_level=2,
                      warmup=False,
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

    cachedir = sparams['cachedir']
    output_dir = sparams['output_dir']

    exp_folder = join(output_dir, 'exp_%s' % job_number)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    debug_folder = join(exp_folder, 'debug')

    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)

    spca = SPARSEPCA
    spca.set_params(
            memory=cachedir,
            dict_init=gparams['dict_init'],
            mask=gparams['mask'],
            smoothing_fwhm=gparams['smoothing_fwhm'],
            alpha=eparams['alpha'],
            random_state=eparams['random_state'],
            feature_ratio=eparams['feature_ratio'],
            debug_folder=debug_folder,
    )

    imgs = gparams['dataset'][slice(*eparams['slice'])]

    train, test = train_test_split(imgs,
                                   random_state=0,
                                   test_size=2)
    spca.fit(train, probe=test)


if __name__ == "__main__":
    nosql_path = sys.argv[3]
    run()
    sqlite3_dumps({" ".join(sys.argv): "JOB DONE"}, nosql_path)
