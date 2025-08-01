# -*- coding: utf-8 -*-
"""GETDIST ROUTINES

Contains interfaces to GetDist routines.
"""

import matplotlib.pyplot as plt
from getdist.mcsamples import loadMCSamples
from getdist import plots
from cloe.auxiliary import yaml_handler
from cloe.auxiliary.run_method import run_is_interactive


def triangle_plot_cobaya(chain_file):
    r"""Produces a triangle plot for the specified chain.

    Parameters
    ----------
    chain_file: str
       Root name for chain files to be passed to GetDist
    """
    sample = loadMCSamples(chain_file, settings={'ignore_rows': 0.3})

    input_path = chain_file + '.input.yaml'
    input = yaml_handler.yaml_read(input_path)

    params = [key for key, value in input['params'].items()
              if isinstance(value, dict)]

    g = plots.get_subplot_plotter(subplot_size=1, width_inch=12, scaling=False)
    g.triangle_plot(sample, params, filled=True, contour_colors=['#FFB300'])
    g.fig.align_ylabels()

    if run_is_interactive():
        plt.show()
    else:
        output_file = chain_file + '.pdf'
        plt.savefig(output_file, dpi=300)
