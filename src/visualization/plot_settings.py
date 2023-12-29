# src/visualization/plot_settings.py

import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style(style='whitegrid', context='notebook', palette='muted', font='Arial', font_scale=1):
    """
    Sets the global plot style for all plots in the project.

    :param style: Style of seaborn plots (e.g., 'whitegrid', 'darkgrid').
    :param context: Context of seaborn plots (e.g., 'notebook', 'paper').
    :param palette: Color palette (e.g., 'muted', 'bright').
    :param font: Default font for all plots.
    :param font_scale: Scaling factor for fonts in plots.
    """
    sns.set_style(style)
    sns.set_context(context)
    sns.set_palette(palette)
    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = font_scale * 10  # Example scaling

def set_figure_size(width=10, height=6):
    """
    Sets the default figure size for matplotlib plots.

    :param width: Width of the figure.
    :param height: Height of the figure.
    """
    plt.rcParams['figure.figsize'] = [width, height]

