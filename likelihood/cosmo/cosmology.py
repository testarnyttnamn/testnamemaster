# -*- coding: utf-8 -*-
"""Cosmology

Module to store cosmological parameters and functions.
"""


def initialiseparamlist():
    """
    Initialises global dictionary to store cosmological parameter values,
    so that they can be assigned and accessed by other modules.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    global cosmoparamdict
    cosmoparamdict = {}
