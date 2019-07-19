import os

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import sqrt

from src.parameters import FIGURE_DIR

# Figure Parameters
MM_TO_INCHES = 1.0 / 25.4
ONE_COLUMN = 89.0 * MM_TO_INCHES
ONE_AND_HALF_COLUMN = 140.0 * MM_TO_INCHES
TWO_COLUMN = 178.0 * MM_TO_INCHES
PAGE_HEIGHT = 247.0 * MM_TO_INCHES
GOLDEN_RATIO = (sqrt(5) - 1.0) / 2.0


def set_figure_defaults():
    # Set background and fontsize
    rc_params = {
        'pdf.fonttype': 42,  # Make fonts editable in Adobe Illustrator
        'ps.fonttype': 42,  # Make fonts editable in Adobe Illustrator
        'axes.labelcolor': '#222222',
        'axes.labelsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'text.color': '#222222',
        'text.usetex': False,
        'figure.figsize': (7.2, 4.45),
        'xtick.major.size': 0.05,
        'ytick.major.size': 0.05,
        'axes.labelpad': 0.1,
    }
    sns.set(style='white', context='paper', rc=rc_params,
            font_scale=1.3)


def save_figure(figure_name, figure_format='pdf'):
    figure_name = os.path.join(FIGURE_DIR, figure_name)
    plt.savefig(f'{figure_name}.{figure_format}', transparent=True,
                dpi=300, bbox_inches='tight')
