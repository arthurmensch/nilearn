import datetime
import glob
import os
import subprocess
import time
from os.path import join

import numpy as np
from clusterlib.scheduler import submit
from clusterlib.storage import sqlite3_loads
from job_lister import build_json_job_list
from lxml.etree import XML, XMLParser

from clusterlib import scheduler

remote_home = '/home/parietal/amensch'
local_home = '/home/parietal/amensch'

script_files = {'exp': join(remote_home, 'work/repos/nilearn/'
                       'examples/sparse_pca/cluster/exp_main.py'),
                'warmup': join(remote_home, 'work/repos/nilearn/'
                          'examples/sparse_pca/cluster/'
                          'warmup_main.py')}

remote_python = subprocess.check_output("""ssh tompouce 'which python'""",
                                        shell=True).strip().decode('ascii')


def _sge_queued_or_running_jobs(user=None, encoding='utf-8'):
    """Get queued or running jobs from SGE queue system."""
    command = ["ssh tompouce 'qstat'", "-xml"]
    if user is not None:
        command.extend(["-u", user])

    try:
        with open(os.devnull, 'w') as shutup:
            xml = subprocess.check_output(command, stderr=shutup)
            tree = XML(xml, parser=XMLParser(encoding=encoding))
            return [leaf.text for leaf in tree.iter("JB_name")]
    except (OSError, subprocess.CalledProcessError):
        # OSError is raised if the program is not installed
        # A CalledProcessError is raised if there is an issue during
        # the call of the command. This might happens if the option -xml
        # is not available such as on rock roll clusters which provide
        # a proxy to qstat whenever only SLURM is installed.
        return []

scheduler._sge_queued_or_running_jobs = _sge_queued_or_running_jobs

queued_or_running_jobs = scheduler.queued_or_running_jobs

def queue_phase(job_dir, phase):
    db_path = join(job_dir, '%s_job.sqlite3' % phase)
    scheduled_jobs = set(queued_or_running_jobs())
    done_jobs = sqlite3_loads(db_path)
    script_file = script_files[phase]
    script_file = script_file.replace(local_home, remote_home)

    n_exp = len(glob.glob(join(job_dir, '%s_*.json' % phase)))
    for exp_json in range(n_exp):
        job_name = "%s%s" % (phase, exp_json)

        job_command = "%s %s %s %s %s" % (remote_python, script_file,
                                          job_dir,
                                          exp_json, db_path)

        if job_name not in scheduled_jobs and job_command not \
                in done_jobs:
            script = submit(job_command, job_name=job_name,
                            time='24:00:00',
                            memory=17000, backend='sge')
            # Remote execution
            script = script.replace('qsub', """ssh tompouce 'qsub'""")
            # os.system(script)
            output = subprocess.check_output(script,
                                             stderr=subprocess.STDOUT,
                                             shell=True)
            print(output)


# Basically a blocking semaphore
def join_phase(job_dir, phase, timeout=None):
    t0 = time.time()
    while True:
        n_jobs = len(glob.glob(join(job_dir, '%s_*.json' % phase)))
        db_path = join(job_dir, '%s_job.sqlite3' % phase)
        done_jobs = sqlite3_loads(db_path)
        if len(done_jobs) == n_jobs:
            return
        if timeout is not None and time.time() - t0 < timeout:
            return
        output = set(queued_or_running_jobs())
        print('qstat:\n %s' % output)
        time.sleep(20)


def queue_jobs(job_dir):
    # queue_phase(job_dir, 'warmup')
    # join_phase(job_dir, 'warmup')
    queue_phase(job_dir, 'exp')
    join_phase(job_dir, 'exp')

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
                        dataset='adhd',
                        output_dir=join(remote_run_dir, 'results'),
                        data_dir=join(remote_home, 'data'),
                        cachedir=join(remote_home, 'nilearn_cache'),
                        alpha_list=np.logspace(-5, 0, 6),
                        feature_ratio_list=np.linspace(1, 10, 5),
                        n_runs=5,
                        n_subjects=77,
                        warmup_slices=20)
    queue_jobs(job_dir)


if __name__ == '__main__':
    main()
