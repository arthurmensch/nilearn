import fnmatch
import json
import os
from os.path import join, expanduser

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.legend as mlegend
from joblib import Parallel
from joblib import delayed
from matplotlib.colors import hsv_to_rgb

from nilearn.decomposition.base import explained_variance, mask_and_reduce
from nilearn.input_data import MultiNiftiMasker
from nilearn_sandbox.datasets import fetch_hcp_rest


def gather_results(job_dir, output_dir):
    full_dict_list = []
    for dirpath, dirname, filenames in os.walk(job_dir):
        for filename in fnmatch.filter(filenames, 'exp_*.json'):
            exp_num, ext = os.path.splitext(filename)
            with open(join(dirpath, filename), 'r') as f:
                exp_dict = json.load(f)
                exp_dict['score_test_file'] = join(output_dir, exp_num,
                                                   'debug/score_test.npy')
                exp_dict['score_test'] = np.load(exp_dict['score_test_file'])
                exp_dict['slice'] = str(exp_dict['slice'])

                full_dict_list.append(exp_dict)

    results = pd.DataFrame(full_dict_list, columns=['feature_ratio',
                                                    'alpha',
                                                    'slice',
                                                    'random_state',
                                                    'score_test',
                                                    # 'density',
                                                    'score_test_file',
                                                    'density_file'])

    results.sort_values(by=['feature_ratio',
                            'alpha',
                            'slice',
                            'random_state'], inplace=True)
    results.to_csv(join(output_dir, 'results.csv'))


def density(job_dir, output_dir):
    masker = MultiNiftiMasker(mask_img=expanduser('~/data/'
                                                  'HCP_mask/mask_img.nii.gz'),
                              standardize=True, detrend=True, smoothing_fwhm=4,
                              )
    neutral_masker = MultiNiftiMasker(mask_img=expanduser('~/data/'
                                                          'HCP_mask/mask_img.nii.gz'),
                                      standardize=False, detrend=False,
                                      smoothing_fwhm=None,
                                      ).fit()
    masker.fit()
    dataset = fetch_hcp_rest(expanduser('~/data'), n_subjects=300)
    test_imgs = dataset.func[1000:1004:4]
    X = mask_and_reduce(masker, test_imgs, reduction_mene)

    for dirpath, dirname, filenames in os.walk(job_dir):
        for filename in fnmatch.filter(filenames, 'exp_*.json'):
            exp_num, ext = os.path.splitext(filename)
            with open(join(dirpath, filename), 'r') as f:
                exp_dict = json.load(f)
            intermediary = join(output_dir, exp_num, 'debug', 'intermediary')
            print('Entering intermediary dir')
            dirfiles = os.listdir(intermediary)
            res = Parallel(n_jobs=24, verbose=10)(delayed(
                    compute_exp_var)(X, neutral_masker, intermediary, filename)
                                                  for filename in
                                                  fnmatch.filter(dirfiles,
                                                                 '*.nii.gz'))

            records, exp_vars, densities = zip(*res)
            exp_dict['records'] = records
            exp_dict['exp_vars'] = exp_vars
            exp_dict['densities'] = densities
            with open(join(dirpath, exp_num + '_stat.json'), 'w+') as f:
                json.dump(exp_dict, f)


def compute_exp_var(X, masker, intermediary, filename):
    record = filename[3:-7]
    print('Computing explained variance')
    components = masker.transform(join(intermediary, filename))
    densities = np.sum(components != 0) / components.shape[1] / \
                components.shape[0]
    exp_var = explained_variance(X, components,
                                 per_component=False)
    return record, exp_var.flat[0], densities


def display_explained_variance(output_dir):
    output_dir = expanduser(output_dir)
    df = pd.read_csv(join(output_dir, 'results.csv'),
                     index_col=list(range(1, 5)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.7)

    feature_ratios = df.index.get_level_values('feature_ratio').unique()
    alphas = df.index.get_level_values('alpha').unique()
    palette = np.array([[352, 1.4, 0.664],
                        [109, 1.228, 1.09],
                        [279, 1.792, 0.715],
                        [204, 0.532, 1.042],
                        [280, 0.43, 0.331],
                        [63, 0.869, 0.864]])
    H = {alpha: h for alpha, h in zip(alphas, palette[:, 0] / 360)}
    V = {feature_ratio: v for
         feature_ratio, v in zip(feature_ratios,
                                 np.linspace(.2, 1, len(feature_ratios)))}
    for index, score in df.ix[:, 'score_test_file'].iteritems():
        score = np.load(score)
        ax.plot(score[:, 0] / index[0],
                score[:, 1],
                marker='+', markevery=5, markersize=3,
                color=hsv_to_rgb(
                        [H[index[1]], V[index[0]], 1 - V[index[0]] / 2]),
                label='Reduc. : %.1f' % index[0])
        ax.set_xlabel('Records')
        ax.set_ylabel('Objective functioon on test set')
        ax.set_title('HCP dataset')
    h, l = ax.get_legend_handles_labels()
    ax.grid(axis='x')
    legend_ratio = mlegend.Legend(ax, h[::36], l[::36], loc='lower left',
                                  bbox_to_anchor=(1, 0))
    alpha_list = ['alpha = %.0e' % alpha for alpha in alphas]
    legend_alpha = mlegend.Legend(ax, h[144:180:6], alpha_list,
                                  loc='upper left',
                                  bbox_to_anchor=(1, 1))
    ax.add_artist(legend_ratio)
    ax.add_artist(legend_alpha)
    fig.savefig(join(output_dir, 'results.pdf'),
                bbox_extra_artists=(legend_alpha, legend_ratio))


def display_explained_variance_cluster(job_dir):
    job_dir = expanduser(job_dir)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.7)

    stat = []
    for filename in fnmatch.filter(os.listdir(job_dir), 'results_stat.json'):
        with open(join(job_dir, filename), 'r') as f:
            stat.append(json.load(f))
    palette = np.array([[139, 203, 75],
                        [167, 91, 180],
                        [192, 77, 75],
                        [155, 165, 190],
                        [189, 151, 73],
                        [77, 63, 61],
                        [123, 189, 145]])
    np.random.shuffle(palette)
    alphas = np.logspace(-5, 0, 6)
    feature_ratios = np.linspace(1, 10, 5)
    H = {alpha: h for alpha, h in zip(alphas, palette[:6, 0] / 360)}
    V = {feature_ratio: v for
         feature_ratio, v in zip(feature_ratios,
                                 np.linspace(0.2, 1, len(feature_ratios)))}
    h_feature_ratios = []
    h_alphas = []
    for this_stat in stat:
        print(float(this_stat['alpha']))
        if True: # this_stat['feature_ratio'] in [1, 5.5, 10]:
            if this_stat['alpha'] in [1e-4, 1e-5]:
                color = hsv_to_rgb([H[this_stat['alpha']],
                                    V[this_stat['feature_ratio']],
                                    1 - V[this_stat['feature_ratio']] / 2])
                # records = list(map(int, this_stat['records']))
                # h, = ax.plot(np.array(records) / this_stat['feature_ratio'],
                #              this_stat['exp_vars'], color=color)
                h, = ax.plot(this_stat['exp_vars'][1::3], this_stat['densities'][1::3],
                                color=color, marker='o', markersize=2)
                h = ax.scatter(this_stat['exp_vars'][-1], this_stat['densities'][-1],
                                color=color, marker='o', s=20)
                if this_stat['random_state'] == 0 and this_stat['slice'] == [0, 394,
                                                                             1]:
                    if this_stat['alpha'] == 1e-4:
                        h_feature_ratios.append(
                                (h, 'F. ratio : %.2f' % this_stat['feature_ratio']))
                    if this_stat['feature_ratio'] == 10:
                        h_alphas.append((h, 'Reg: %.0e' % this_stat['alpha']))
    ax.set_xlabel('Density')
    ax.set_ylabel('Explained variance on test set')
    ax.set_title('HCP dataset')
    ax.grid(axis='x')
    legend_ratio = mlegend.Legend(ax, *list(zip(*h_feature_ratios)),
                                  loc='lower left',
                                  bbox_to_anchor=(1, 0))
    legend_alpha = mlegend.Legend(ax, *list(zip(*h_alphas)),
                                  loc='upper left',
                                  bbox_to_anchor=(1, 1))
    ax.add_artist(legend_ratio)
    ax.add_artist(legend_alpha)
    ax.set_xlim([0, 0.12])
    ax.set_ylim([0, 0.25])
    fig.savefig(join(job_dir, 'results.pdf'),
                bbox_extra_artists=(legend_alpha, legend_ratio))


if __name__ == '__main__':
    # density(expanduser("~/share/output/spca_cluster/2016-01-11_15-32-09/jobs"),
    #         expanduser(
    #             "~/share/output/spca_cluster/2016-01-11_15-32-09/results"))
    display_explained_variance_cluster(
            "~/jobs")
