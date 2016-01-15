import fnmatch
import json
import os
from os.path import join, expanduser

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.legend as mlegend
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
                                                  'HCP_mask/mask_img.nii.gz'))
    masker.fit()
    dataset = fetch_hcp_rest(expanduser('~/data'))
    test_imgs = dataset.func[1000:1002]
    X = mask_and_reduce(test_imgs, masker=masker, reduction_method=None)

    full_dict_list = []
    for dirpath, dirname, filenames in os.walk(job_dir):
        for filename in fnmatch.filter(filenames, 'exp_*.json'):
            exp_num, ext = os.path.splitext(filename)
            with open(join(dirpath, filename), 'r') as f:
                exp_dict = json.load(f)
            intermediary = join(output_dir, exp_num, 'intermediary')
            records = []
            exp_vars = []
            densities = []
            for filename in os.listdir(intermediary):
                records.append(filename[3:-7])
                components = masker.transform(filename)
                densities.append(
                        np.sum(components != 0) / components.flat.shape[0])
                exp_vars.append(explained_variance(X, components,
                                                   per_component=False))
            exp_dict['records'] = records
            exp_dict['exp_vars'] = exp_vars
            exp_dict['densities'] = densities
            with open(join(dirpath, filename), 'w+') as f:
                json.dump(exp_dict, f)

def display_explained_variance(output_dir):
    output_dir = expanduser(output_dir)
    df = pd.read_csv(join(output_dir, 'results.csv'),
                     index_col=list(range(1, 5)))
    fig = plt.figure()
    # gs = gridspec.GridSpec(1, 2, width_ratios=[5, 2])
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.7)
    # ax_legend = fig.add_subplot(111)
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
                color=hsv_to_rgb([H[index[1]], V[index[0]], 1 - V[index[0]] / 2]),
                label='Reduc. : %.1f' % index[0])
        ax.set_xlabel('Records')
        ax.set_ylabel('Objective functioon on test set')
        ax.set_title('HCP dataset')
    h, l = ax.get_legend_handles_labels()
    ax.grid(axis='x')
    legend_ratio = mlegend.Legend(ax, h[::36], l[::36], loc='lower left',
                                  bbox_to_anchor=(1, 0))
    alpha_list = ['alpha = %.0e' % alpha for alpha in alphas]
    legend_alpha = mlegend.Legend(ax, h[144:180:6], alpha_list, loc='upper left',
                                  bbox_to_anchor=(1, 1))
    ax.add_artist(legend_ratio)
    ax.add_artist(legend_alpha)
    fig.savefig(join(output_dir, 'results.pdf'),
                bbox_extra_artists=(legend_alpha, legend_ratio))


if __name__ == '__main__':
    # gather_results(expanduser("~/share/output/spca_cluster/2016-01-11_15-32-09/jobs"),
    #                expanduser("~/share/output/spca_cluster/2016-01-11_15-32-09/results"))
    display_explained_variance(
        "~/share/output/spca_cluster/2016-01-11_15-32-09/results")
