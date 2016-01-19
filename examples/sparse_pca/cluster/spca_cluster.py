import datetime
import glob
import os
import subprocess
import time
from os.path import join

import numpy as np
from clusterlib.scheduler import submit, queued_or_running_jobs
from clusterlib.storage import sqlite3_loads
from job_lister import build_json_job_list

# from lxml.etree import XML, XMLParser
#
# from clusterlib import scheduler
from joblib import Parallel, delayed

remote_home = '/home/parietal/amensch'
local_home = '/home/parietal/amensch'

script_files = {'exp': join(remote_home, 'work/repos/nilearn/'
                       'examples/sparse_pca/cluster/exp_main.py'),
                'warmup': join(remote_home, 'work/repos/nilearn/'
                          'examples/sparse_pca/cluster/'
                          'warmup_main.py')}

remote_python = subprocess.check_output("""which python """,
                                        shell=True).strip().decode('ascii')

def queue_phase(job_dir, phase):
    db_path = join(job_dir, '%s_job.sqlite3' % phase)
    scheduled_jobs = set(queued_or_running_jobs())
    done_jobs = sqlite3_loads(db_path)
    script_file = script_files[phase]
    script_file = script_file.replace(local_home, remote_home)

    n_exp = len(glob.glob(join(job_dir, '%s_*.json' % phase)))
    for exp_json in range(n_exp):
        job_name = "%s_%s" % (phase, exp_json)

        job_command = "%s %s %s %s %s" % (remote_python, script_file,
                                          job_dir,
                                          exp_json, db_path)

        if job_name not in scheduled_jobs and job_command not \
                in done_jobs:
            script = submit(job_command, job_name=job_name,
                            log_directory=job_dir,
                            time='72:00:00',
                            memory=20000, backend='sge')
            output = subprocess.check_output(script,
                                             shell=True)


# Basically a blocking semaphore
def join_phase(job_dir, phase, timeout=None):
    t0 = time.time()
    n_jobs = len(glob.glob(join(job_dir, '%s_*.json' % phase)))
    db_path = join(job_dir, '%s_job.sqlite3' % phase)
    done_jobs = sqlite3_loads(db_path)
    if len(done_jobs) == n_jobs:
        return
    if timeout is not None and time.time() - t0 < timeout:
        return
    queue = set(queued_or_running_jobs())
    print('qstat:\n %s' % queue)
    time.sleep(120)
    return queue


def queue_jobs(job_dir):
    while True:
        queue_phase(job_dir, 'exp')
        queue = join_phase(job_dir, 'exp')
        if len(queue) == 0:
            return


def main():
    # make sure we run the same nilearn on cluster / local
    # Should use virtualenv there, for generic code

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    local_run_dir = join(local_home, 'share/output/spca_cluster', timestamp)
    remote_run_dir = join(remote_home, 'share/output/spca_cluster', timestamp)
    if not os.path.exists(local_run_dir):
        os.makedirs(local_run_dir)
    job_dir = join(local_run_dir, 'jobs')
    build_json_job_list(job_dir,
                        dataset='hcp',
                        output_dir=join(remote_run_dir, 'results'),
                        data_dir=join(remote_home, 'data'),
                        cachedir=join(remote_home, 'nilearn_cache'),
                        alpha_list=np.logspace(-4, -1, 4),
                        feature_ratio_list=np.linspace(1, 10, 5),
                        n_runs=2,
                        n_slices=1,
                        n_subjects=100,
                        warmup_slices=80)
    queue_jobs(job_dir)


if __name__ == '__main__':
    main()
