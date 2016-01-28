import datetime
import fnmatch
import glob
import json
import os
from os.path import expanduser, join

import matplotlib.legend as mlegend
import matplotlib.pyplot as plt
import numpy as np
from joblib import delayed, Parallel
from matplotlib import gridspec
from matplotlib.colors import hsv_to_rgb

from nilearn.datasets import fetch_atlas_smith_2009
from nilearn.decomposition.base import explained_variance
from nilearn.decomposition.dict_fact import DictMF
from nilearn.decomposition.sparse_pca import objective_function
from nilearn.input_data import MultiNiftiMasker
from sklearn.utils import check_random_state

import seaborn.apionly as sns


def load_data(init='rsn70',
              dataset='hcp'):
    if dataset == 'hcp':
        with open(expanduser('~/data/HCP_unmasked/data.json'), 'r') as f:
            data = json.load(f)
        for this_data in data:
            this_data['array'] += '.npy'
        mask_img = expanduser('~/data/HCP_mask/mask_img.nii.gz')
    elif dataset == 'adhd':
        with open(expanduser('~/data/ADHD_unmasked/data.json'), 'r') as f:
            data = json.load(f)
        mask_img = expanduser('~/data/ADHD_mask/mask_img.nii.gz')
    masker = MultiNiftiMasker(mask_img=mask_img, smoothing_fwhm=4,
                              standardize=True)
    masker.fit()
    smith2009 = fetch_atlas_smith_2009()
    if init == 'rsn70':
        init = smith2009.rsn70
    elif init == 'rsn20':
        init = smith2009.rsn20
    dict_init = masker.transform(init)
    return masker, dict_init, sorted(data, key=lambda t: t['img'])


def compute_exp_var(X, masker, filename, alpha=None):
    print('Computing explained variance')
    components = masker.transform(filename)
    densities = np.sum(components != 0) / components.size
    if alpha is None:
        exp_var = explained_variance(X, components,
                                     per_component=False).flat[0]
    else:
        exp_var = objective_function(X, components, alpha)
    return exp_var, densities


def analyse_exp_var_in_dir(output_dir, dataset='hcp', objective=False):
    masker, _, data = load_data(dataset=dataset)

    if dataset == 'hcp':
        data = data[400:404:4]
    elif dataset == 'adhd':
        data = data[36:]

    masker.set_params(smoothing_fwhm=None, standardize=False)
    with open(join(output_dir, 'experiment.json'), 'r') as f:
        exp_dict = json.load(f)
    concatenated_data = [np.load(this_data['array']) for this_data in data]
    X = np.concatenate(concatenated_data, axis=0)
    output_files = os.listdir(output_dir)
    records = []
    exp_vars = []
    densities = []
    for filename in sorted(fnmatch.filter(output_files, 'a_*.nii.gz'))[
                    ::exp_dict['reduction']]:
        exp_var, density = compute_exp_var(X, masker,
                                           join(output_dir, filename),
                                           alpha=exp_dict['alpha'] if objective
                                           else None)
        records.append(int(filename[2:-7]))
        exp_vars.append(exp_var)
        densities.append(density)
        exp_dict['records'] = records
        if objective:
            exp_dict['objective'] = exp_vars
        else:
            exp_dict['exp_vars'] = exp_vars
        exp_dict['densities'] = densities
        with open(join(output_dir, 'experiment.json'), 'w+') as f:
            json.dump(exp_dict, f)
    order = np.argsort(np.array(records))
    exp_vars = np.array(exp_vars)[order].tolist()
    densities = np.array(densities)[order].tolist()
    records = np.array(records)[order].tolist()
    exp_dict['records'] = records
    if objective:
        exp_dict['objective'] = exp_vars
    else:
        exp_dict['exp_vars'] = exp_vars
    exp_dict['densities'] = densities
    with open(join(output_dir, 'experiment_stat.json'), 'w+') as f:
        json.dump(exp_dict, f)


def single(output_dir, alpha, reduction, impute, dataset, init, records_range,
           random_state=0):
    masker, dict_init, data = load_data(dataset=dataset, init=init)
    n_components = dict_init.shape[0]
    dict_mf = DictMF(batch_size=20, reduction=reduction,
                     random_state=random_state,
                     learning_rate=1,
                     dict_init=dict_init,
                     alpha=alpha,
                     impute=impute,
                     l1_ratio=.5,
                     fit_intercept=False,
                     n_components=n_components,
                     backend='python',
                     debug=True,
                     )
    random_state = check_random_state(0)
    data = [data[i] for i in records_range]
    for e in range(5):
        random_state.shuffle(data)
        for i, this_data in enumerate(data):
            X = np.load(this_data['array'])
            dict_mf.partial_fit(X)
            print('Loaded record %i, '
                  ' seen rows: %i' % (i, dict_mf.counter_[0]))
            if i % 5 == 0:
                density = np.sum(dict_mf.Q_ != 0) / dict_mf.Q_.size
                print('Red. %.2f, '
                      'dictionary density: %.4f' % (dict_mf.reduction,
                                                    density))
                if density < 1e-3:
                    print('Dictionary is too sparse, giving up')
                    return
                components = masker.inverse_transform(dict_mf.Q_)
                components.to_filename(join(output_dir, 'a_%i.nii.gz'
                                            % dict_mf.counter_[0]))
                if dict_mf.debug:
                    with open(join(output_dir, 'loss.json'), 'w+') as f:
                        json.dump(dict_mf.loss_, f)


def launch_from_dir(output_dir):
    with open(join(output_dir, 'experiment.json'), 'r') as f:
        exp_dict = json.load(f)
    single(output_dir, exp_dict['alpha'], exp_dict['reduction'],
           exp_dict['impute'], exp_dict['dataset'], exp_dict['init'],
           exp_dict['records_range'], exp_dict['random_state'])


def main(dataset='hcp', init='rsn70', n_jobs=1):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                 '-%M-%S')
    output_dir = expanduser('~/output/fast_spca/%s' % timestamp)

    os.makedirs(output_dir)

    # alphas = np.logspace(-6, -2, 5)
    # reductions = np.linspace(1, 9, 5)
    alphas = [0.01]
    reductions = [1, 3]
    random_states = {1: [0], 3: list(range(2))}
    imputes = [False]

    if dataset == 'hcp':
        records_range = np.arange(400)
    elif dataset == 'adhd':
        records_range = np.arange(36)
    records_range = records_range.tolist()

    i = 0
    experiment_dirs = []
    for alpha in alphas:
        for reduction in reductions:
            for impute in imputes:
                for this_random_state in random_states[reduction]:
                    experiment = {}
                    experiment_dir = join(output_dir, 'experiment_%i' % i)
                    experiment_dirs.append(experiment_dir)
                    os.makedirs(experiment_dir)
                    experiment['alpha'] = alpha
                    experiment['reduction'] = reduction
                    experiment['impute'] = impute
                    experiment['dataset'] = dataset
                    experiment['init'] = init
                    experiment['records_range'] = records_range
                    experiment['random_state'] = this_random_state
                    print(experiment)
                    with open(join(output_dir, 'experiment_%i' % i,
                                   'experiment.json'),
                              'w+') as f:
                        json.dump(experiment, f)
                    i += 1

    Parallel(n_jobs=n_jobs)(delayed(launch_from_dir)(experiment_dir)
                            for experiment_dir in experiment_dirs)


def simple(alpha, reduction, impute, dataset, init, records_range):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                 '-%M-%S')
    output_dir = expanduser('~/output/fast_spca/%s' % timestamp)

    os.makedirs(output_dir)

    single(output_dir, alpha, reduction, impute, dataset, init, records_range,
           random_state=0)


def analyse_dir(output_dir, dataset='hcp', objective=False, n_jobs=1):
    experiment_dirs = fnmatch.filter(os.listdir(output_dir), 'experiment_*')
    Parallel(n_jobs=n_jobs)(
            delayed(analyse_exp_var_in_dir)(join(output_dir, experiment_dir),
                                            objective=objective,
                                            dataset=dataset)
            for experiment_dir in experiment_dirs)


def analyse_distance(output_dir, dataset='hcp'):
    masker, _, _ = load_data(dataset=dataset)

    masker.set_params(smoothing_fwhm=None, standardize=False)

    dictionaries = {}
    records = {}

    experiment_dirs = fnmatch.filter(os.listdir(output_dir), 'experiment_*')
    min_len = 10000
    for exp in experiment_dirs:
        output_exp = join(output_dir, exp)
        output_files = os.listdir(output_exp)
        dictionaries[exp] = []
        records[exp] = []
        for filename in fnmatch.filter(output_files, 'a_*.nii.gz'):
            dictionaries[exp].append(masker.transform(join(output_dir, output_exp,
                                                           filename)))
            records[exp].append(int(filename[2:-7]))

        records[exp] = np.array(records[exp])
        order = records[exp].argsort()
        records[exp] = records[exp][order]
        dictionaries[exp] = np.array(dictionaries[exp])
        dictionaries[exp] = dictionaries[exp][order]
        min_len = min(len(dictionaries[exp]), min_len)
    ref_dict = dictionaries.pop('experiment_0')[:min_len]
    dictionaries = [dictionary[:min_len] for dictionary in dictionaries.values()]
    dictionaries = np.array(dictionaries)
    mean_dict = dictionaries.mean(axis=0)
    diff_norm = np.sum((mean_dict - ref_dict) ** 2, axis=(1, 2))
    var_dict = (dictionaries - ref_dict).var(axis=0)
    var_norm = np.sum(var_dict, axis=(1, 2))
    results = dict(diff_norm=diff_norm.tolist(), var_norm=var_norm.tolist(),
                   records=records[0][:len(mean_dict)].tolist())
    json.dump(results, open(join(output_dir, 'diff.json'), 'w+'))


def plot_diff(output_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    results = json.load(open(join(output_dir, 'diff.json'), 'r'))
    ax.plot(results['records'], results['diff'])
    plt.savefig(join(output_dir, 'diff.pdf'))


def display_explained_variance_density(output_dir, impute=True):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    # ax_pareto = fig.add_subplot(gs[3])
    fig.subplots_adjust(right=0.8)
    fig.subplots_adjust(bottom=0.2)

    stat = []
    alphas = []
    reductions = []
    for filename in glob.glob(join(output_dir, '**/experiment_stat.json'),
                              recursive=True):
        with open(filename, 'r') as f:
            print(filename)
            stat.append(json.load(f))
        alphas.append(stat[-1]['alpha'])
        reductions.append(stat[-1]['reduction'])
    alphas = np.unique(np.array(alphas))
    reductions = np.unique(np.array(reductions))
    H = {alpha: h for alpha, h in
         zip(alphas, np.linspace(0, 360 - 360 / len(alphas),
                                 len(alphas)) / 360)}
    V = {reduction: v for
         reduction, v in zip(reductions, np.linspace(0.2, 1, len(reductions)))}
    h_reductions = []
    h_alphas = []
    pareto_fronts = {reduction: [] for reduction in reductions}
    min_len = 10000
    for this_stat in stat:
        if len(this_stat['records']) > 0 and this_stat['impute'] == impute:
            min_len = min(min_len, len(this_stat['objective']))
    min_len -= 1
    min_len = -1
    ax = {}
    for i, alpha in enumerate([1e-2, 1e-3, 1e-4]):
        ax[alpha] = fig.add_subplot(gs[i])
        if i == 0:
            ax[alpha].set_ylabel('Objective value on test set')
        ax[alpha].set_title('$\\alpha$  = %.0e' % alpha)
        ax[alpha]
        # ax[alpha].grid()
        ax[alpha].set_xscale('log')
    colormap = sns.cubehelix_palette(9, start=0, rot=0, dark=.3, light=.8,
                                     reverse=False)
    colormap[0] = [1, .8, .8]
    # colormap = sns.color_palette("Blues", 9)
    for this_stat in stat:
        if len(this_stat['records']) > 0 and this_stat['impute'] == impute \
                and this_stat['alpha'] in [1e-2, 1e-3, 1e-4]:

            print("%s %s" % (this_stat['alpha'], this_stat['reduction']))
            # pareto_fronts[this_stat['reduction']].append(
            #         [1 - this_stat['densities'][min_len],
            #          this_stat['objective'][min_len]])
            # s = ax_pareto.scatter(1 - this_stat['densities'][min_len],
            #                this_stat['objective'][min_len],
            #                color=color, marker='o', s=20)
            s, = ax[this_stat[
                'alpha']].plot(np.array(this_stat['records']) /
                               this_stat['reduction'] / (1200 *
                                                         400),
                               this_stat['objective'],
                               color=colormap[int(this_stat[
                                                      'reduction']) - 1],
                               marker='o',
                               markersize=1)
            if this_stat['alpha'] == 1e-4:
                h_reductions.append(
                        (s, '%.2f' % this_stat['reduction']))
                # if this_stat['reduction'] == 7:
                #     h_alphas.append((h, '%.0e' % this_stat['alpha']))
                # for reduction, pareto_front in pareto_fronts.items():
                #     print(reduction)
                #     color = [1 - V[reduction] / 2] * 3
                #     values = np.array(list(zip(*pareto_front)))
                #     order = np.argsort(values[0])
                # ax_pareto.plot(values[0, order], values[1, order], color=color)
    # ax.set_title('ADHD dataset, impute = %r' % impute)
    # ax_pareto.grid()
    # ax_pareto.set_xlabel('Dictionary sparsity')
    # ax_time.set_ylabel('Explained variance on test set')
    legend_ratio = mlegend.Legend(ax[1e-4], *list(zip(*h_reductions)),
                                  loc='lower left',
                                  bbox_to_anchor=(1.05, -.15),
                                  title='Reduction')
    # legend_alpha = mlegend.Legend(ax[10e-4], *list(zip(*h_alphas)),
    #                               loc='upper left',
    #                               bbox_to_anchor=(1.05, 1),
    #                               title='Regularisation')
    ax[1e-4].add_artist(legend_ratio)
    # ax_pareto.add_artist(legend_alpha)
    # ax_pareto.set_xlim([0.8, 1])
    # ax_pareto.set_xticks([0.8, 0.9, 1])

    fig.savefig(join(output_dir, 'results_%r.pdf' % impute),
                bbox_extra_artists=(legend_ratio))


if __name__ == '__main__':
    main('adhd', 'rsn20', n_jobs=3)
    # analyse_distance('/home/arthur/output/fast_spca/2016-01-28_18-06-42', 'adhd')
    # plot_diff('/media/storage/output/fast_spca/2016-01-28_17-16-23')
    # simple(1e-4, 3, True, 'adhd', 'rsn20', list(range(0, 36)))
    # analyse_dir('/storage/workspace/amensch/output/fast_spca/2016-01-26_15-31-43',   n_jobs=20, objective=True)
    # display_explained_variance_density(
    #         expanduser('~/drago/output/fast_spca/2016-01-26_15-31-43'),
    #         impute=False)
    # display_explained_variance_density('/home/parietal/amensch/output/fast_spca/2016-01-25_23-56-39')
    # display_explained_variance_density(
    #     expanduser('~/drago/output/fast_spca/2016-01-25_23-56-39'),
    #     impute=False)
    # display_explained_variance_density(
    #     expanduser('/home/arthur/drago/output/fast_spca/2016-01-25_22-21-50'),
    #     impute=False)
