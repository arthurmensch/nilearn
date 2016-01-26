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
from matplotlib.colors import hsv_to_rgb

from nilearn.datasets import fetch_atlas_smith_2009
from nilearn.decomposition.base import explained_variance
from nilearn.decomposition.dict_fact import DictMF
from nilearn.input_data import MultiNiftiMasker
from sklearn.utils import check_random_state


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


def compute_exp_var(X, masker, filename):
    print('Computing explained variance')
    components = masker.transform(filename)
    densities = np.sum(components != 0) / components.size
    exp_var = explained_variance(X, components,
                                 per_component=False)
    return exp_var.flat[0], densities


def analyse_exp_var_in_dir(output_dir, dataset='hcp'):
    masker, _, data = load_data(dataset=dataset)

    if dataset == 'hcp':
        data = data[400:404:4]
    elif dataset == 'adhd':
        data = data[36:]

    masker.set_params(smoothing_fwhm=None, standardize=False)
    with open(join(output_dir, 'experiment.json'), 'r') as f:
        exp_dict = json.load(f)
    concatenated_data = [np.load(this_data['array']) for this_data in
            data[36:]]
    X = np.concatenate(concatenated_data, axis=0)
    output_files = os.listdir(output_dir)
    records = []
    exp_vars = []
    densities = []
    for filename in fnmatch.filter(output_files, 'a_*.nii.gz'):
        exp_var, density = compute_exp_var(X, masker,
                                           join(output_dir, filename))
        records.append(int(filename[2:-7]))
        exp_vars.append(exp_var)
        densities.append(density)
        exp_dict['records'] = records
        exp_dict['exp_vars'] = exp_vars
        exp_dict['densities'] = densities
        with open(join(output_dir, 'experiment.json'), 'w+') as f:
            json.dump(exp_dict, f)
    order = np.argsort(np.array(records))
    exp_vars = np.array(exp_vars)[order]
    densities = np.array(densities)[order]
    records = np.array(records)[order]
    exp_dict['records'] = records.tolist()
    exp_dict['exp_vars'] = exp_vars.tolist()
    exp_dict['densities'] = densities.tolist()
    with open(join(output_dir, 'experiment.json'), 'w+') as f:
        json.dump(exp_dict, f)


def single(output_dir, alpha, reduction, impute, dataset, init, records_range):
    masker, dict_init, data = load_data(dataset=dataset, init=init)
    n_components = dict_init.shape[0]
    dict_mf = DictMF(batch_size=20, reduction=reduction, random_state=0,
                     learning_rate=1,
                     dict_init=dict_init,
                     alpha=alpha,
                     impute=impute,
                     l1_ratio=.5,
                     fit_intercept=False,
                     n_components=n_components,
                     backend='python',
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
                    fig = plt.figure()
                    plt.plot(np.arange(len(dict_mf.loss_[1:])), dict_mf.loss_[1:])
                    plt.savefig(join(output_dir, 'loss.pdf'))
                    plt.close(fig)


def launch_from_dir(output_dir):
    with open(join(output_dir, 'experiment.json'), 'r') as f:
        exp_dict = json.load(f)
    single(output_dir, exp_dict['alpha'], exp_dict['reduction'],
           exp_dict['impute'], exp_dict['dataset'], exp_dict['init'],
           exp_dict['records_range'])


def main(dataset='hcp', init='rsn70', n_jobs=1):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                 '-%M-%S')
    output_dir = expanduser('~/output/fast_spca/%s' % timestamp)

    os.makedirs(output_dir)

    alphas = np.logspace(-4, -1, 4)
    reductions = np.linspace(1, 7, 4)
    imputes = [False]

    if dataset == 'hcp':
        records_range = np.arange(400)
    elif dataset == 'adhd':
        records_range = np.arange(36)
    records_range  = records_range.tolist()

    i = 0
    experiment_dirs = []
    for alpha in alphas:
        for reduction in reductions[::-1]:
            for impute in imputes:
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
                print(experiment)
                with open(join(output_dir, 'experiment_%i' % i, 'experiment.json'),
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

    single(output_dir, alpha, reduction, impute, dataset, init, records_range)


def analyse_dir(output_dir, dataset='hcp', n_jobs=1):
    experiment_dirs = fnmatch.filter(os.listdir(output_dir), 'experiment_*')
    Parallel(n_jobs=n_jobs)(
            delayed(analyse_exp_var_in_dir)(join(output_dir, experiment_dir),
                                            dataset=dataset)
            for experiment_dir in experiment_dirs)


def display_explained_variance_density(output_dir, impute=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.7)

    stat = []
    alphas = []
    reductions = []
    for filename in glob.glob(join(output_dir, '**/experiment.json'),
                              recursive=True):
        with open(filename, 'r') as f:
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
            min_len = min(min_len, len(this_stat['exp_vars']))
    min_len -= 1
    min_len = -1
    for this_stat in stat:
        if len(this_stat['records']) > 0 and this_stat['impute'] == impute:
            print("%s %s" % (this_stat['alpha'], this_stat['reduction']))
            color = hsv_to_rgb([H[this_stat['alpha']],
                                V[this_stat['reduction']],
                                1 - V[this_stat['reduction']] / 2])
            pareto_fronts[this_stat['reduction']].append(
                    [this_stat['densities'][min_len],
                     this_stat['exp_vars'][min_len]])
            # ax.plot(this_stat['densities'],
            #        this_stat['exp_vars'],
            #        color=color, marker='o', markersize=3)
            h = ax.scatter(this_stat['densities'][min_len],
                           this_stat['exp_vars'][min_len],
                           color=color, marker='o', s=20)
            if this_stat['alpha'] == 1e-4:
                h_reductions.append(
                        (h, 'F. ratio : %.2f' % this_stat['reduction']))
            if this_stat['reduction'] == 7:
                h_alphas.append((h, 'Reg: %.0e' % this_stat['alpha']))
    for reduction, pareto_front in pareto_fronts.items():
        print(reduction)
        color = [1 - V[reduction] / 2] * 3
        values = np.array(list(zip(*pareto_front)))
        order = np.argsort(values[0])
        plt.plot(values[0, order], values[1, order], color=color)

    ax.set_xlabel('Density')
    ax.set_ylabel('Explained variance on test set')
    ax.set_title('ADHD dataset, impute = %r' % impute)
    ax.grid(axis='x')
    legend_ratio = mlegend.Legend(ax, *list(zip(*h_reductions)),
                                  loc='lower left',
                                  bbox_to_anchor=(1, 0))
    legend_alpha = mlegend.Legend(ax, *list(zip(*h_alphas)),
                                  loc='upper left',
                                  bbox_to_anchor=(1, 1))
    ax.add_artist(legend_ratio)
    ax.add_artist(legend_alpha)
    ax.set_xlim([0.0, 0.4])
    # ax.set_ylim([-0.01, 0.15])
    fig.savefig(join(output_dir, 'results_%r.pdf' % impute),
                bbox_extra_artists=(legend_alpha, legend_ratio))


if __name__ == '__main__':
    # main('adhd', 'rsn70')
    # simple(1e-5, 5, True, 'adhd', 'rsn20', list(range(0, 36)))
    analyse_dir('/storage/workspace/amensch/output/fast_spca/2016-01-25_23-56-39',
                n_jobs=36)
    # display_explained_variance_density('/home/parietal/amensch/output/fast_spca/2016-01-25_23-56-39')
    # display_explained_variance_density(expanduser('~/drago/output/fast_spca/2016-01-25_22-21-50'), impute=True)
