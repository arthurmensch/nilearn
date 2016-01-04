import datetime
import glob
import os
import sys
from os.path import expanduser, join

import numpy as np
from clusterlib.scheduler import queued_or_running_jobs
from clusterlib.scheduler import submit
from clusterlib.storage import sqlite3_loads

from job_lister import build_json_job_list

script_files = {'exp': '/volatile/arthur/work/repos/nilearn/examples/sparse_pca/cluster/exp_main.py',
              'warmup': '/volatile/arthur/work/repos/nilearn/examples/sparse_pca/cluster/warmup_main.py'}

def queue_jobs(job_dir):
    nosql_path = join(job_dir, 'job.sqlite3')

    scheduled_jobs = set(queued_or_running_jobs())
    done_jobs = sqlite3_loads(nosql_path)
    for phase in ['warmup', 'exp']:
        script_file = script_files[phase]
        n_exp = len(glob.glob(join(job_dir, '%s_*.json' % phase)))
        for exp_json in range(n_exp):
            job_name = "%s %s" % (phase, exp_json)
            job_command = "%s %s %s %s %s" % (sys.executable, script_file,
                                              job_dir,
                                              exp_json, nosql_path)

            if job_name not in scheduled_jobs and job_command not in done_jobs:
                script = submit(job_command, job_name=job_name)
                print(script)
                os.system(job_command)


def main():
    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = join(expanduser('~/output/spca_cluster'), 'tompouce_0')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    job_dir = join(run_dir, 'jobs')
    build_json_job_list(job_dir,
                        output_dir=join(run_dir, 'results'),
                        alpha_list=np.logspace(-5, 0, 6),
                        feature_ratio_list=np.linspace(1, 10, 5),
                        n_subjects=40,
                        warmup_slices=40)
    queue_jobs(job_dir)


if __name__ == '__main__':
    main()
