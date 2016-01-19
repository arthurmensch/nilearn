import datetime
import glob
import os
import subprocess
import time
from os.path import join, expanduser

import numpy as np
from clusterlib.scheduler import submit, queued_or_running_jobs
from clusterlib.storage import sqlite3_loads
from job_lister import build_json_job_list

# from lxml.etree import XML, XMLParser
#
# from clusterlib import scheduler
from joblib import Parallel, delayed


def queue_phase(job_dir, phase):
    if phase == 'exp':
        from exp_main import run
    else:
        from warmup_main import run

    n_exp = len(glob.glob(join(job_dir, '%s_*.json' % phase)))

    Parallel(n_jobs=32, verbose=10)(
            delayed(run)(['', job_dir, exp_json])
            for exp_json in range(n_exp))


def queue_jobs(job_dir):
    queue_phase(job_dir, 'warmup')
    queue_phase(job_dir, 'exp')


def main():
    # make sure we run the same nilearn on cluster / local
    # Should use virtualenv there, for generic code

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = expanduser(join('~/output/spca_cluster', timestamp))
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    job_dir = join(run_dir, 'jobs')
    build_json_job_list(job_dir,
                        dataset='hcp',
                        output_dir=join(run_dir, 'results'),
                        data_dir=expanduser('~/data'),
                        cachedir=expanduser('~/nilearn_cache'),
                        alpha_list=np.logspace(-4, -1, 4),
                        feature_ratio_list=np.linspace(1, 10, 4),
                        n_runs=2,
                        n_slices=1,
                        n_subjects=100,
                        warmup_slices=24)
    queue_jobs(job_dir)


if __name__ == '__main__':
    main()
