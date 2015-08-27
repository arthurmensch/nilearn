import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from . import plot_stat_map, plot_prob_atlas
from .._utils import check_niimg_4d
from ..image import index_img
from ..datasets import fetch_atlas_smith_2009

def plot_to_pdf(img, path='multipages.pdf', vmax='auto'):
    a4_size = (8.27,11.69)
    img = check_niimg_4d(img)
    n_components = img.shape[3]

    if vmax == 'auto':
        vmax = np.max(np.abs(img.get_data()), axis=3)
        vmax[vmax >= 0.1] = 0
        vmax = np.max(vmax)
    else:
        vmax = float(vmax)
    with PdfPages(path) as pdf:
        for i in range(-1, n_components, 5):
            fig, axes = plt.subplots(5, 1, figsize=a4_size, squeeze=False)
            axes = axes.reshape(-1)
            for j, ax in enumerate(axes):
                if i + j < 0:
                    plot_prob_atlas(img, axes=ax)
                elif j + i < n_components:
                    plot_stat_map(index_img(img, j + i), axes=ax, vmax=vmax)
                else:
                    ax.axis('off')
            pdf.savefig(fig)
            plt.close()


def test_plot_to_pdf():
    smith = fetch_atlas_smith_2009()
    img = smith.rsn10
    plot_to_pdf(img)
