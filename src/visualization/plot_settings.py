# src/visualization/plot_settings.py

import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style(style='whitegrid', context='notebook', palette='muted', font='Arial', font_scale=1):
    """
    Sets the plot style using seaborn and matplotlib.
    :param style: The style to set (seaborn or matplotlib styles).
    :param context: The plotting context parameter (e.g., 'notebook', 'paper').
    :param palette: The color palette to use (e.g., 'muted', 'bright').
    :param font: The font family to use for the plots.
    :param font_scale: Scaling factor for fonts.
    """
    # Check if the style is a seaborn style
    seaborn_styles = ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']
    if style in seaborn_styles:
        sns.set_style(style)
    else:
        plt.style.use(style)
    
    sns.set_context(context, font_scale=font_scale)
    sns.set_palette(palette)
    plt.rcParams['font.family'] = font

def set_figure_size(width=10, height=6):
    """
    Sets the default figure size for matplotlib plots.

    :param width: Width of the figure.
    :param height: Height of the figure.
    """
    plt.rcParams['figure.figsize'] = [width, height]

