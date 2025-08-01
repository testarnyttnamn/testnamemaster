"""OBSERVABLES DEALER

Contains function to read the observable dictionary
and plot the visualization of the matrix.
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings


def observables_visualization(observables_dict, palette='standard'):
    """
    Observables visualisation.

    Takes the observables dictionary read by Cobaya
    and shows the visualisation of the matrix.

    Parameters
    ----------
    observables_dict: dict
        Dictionary with the observables selection
    palette: string
        Name of the visualisation color palette
        Choose between 'standard', 'protanopia',
        'deuteranopia'
    """
    observables_df = pd.DataFrame(
        observables_dict).fillna(-1).astype(int).T

    cmaps = {
        'standard': ('white', '#f4d4d4', '#85c0f9'),
        'protanopia': ('white', '#ae9c45', '#a7b8f8'),
        'deuteranopia': ('white', '#c59434', '#a3b7f9')
    }
    cmap = LinearSegmentedColormap.from_list(palette,
                                             cmaps[palette],
                                             len(cmaps[palette]))
    ax = sns.heatmap(observables_df, annot=False,
                     cmap=cmap,
                     cbar=True)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-0.667, 0, 0.667])
    colorbar.set_ticklabels(['No input', 'False', 'True'])
    plt.title("matrix with observables selection")
    plt.show()
    return observables_df


def observables_selection_checker(observables_dict):
    """
    Observables selection checker.

    Checks and notifies the user if a selection is
    not correct.

    Parameters
    ----------
    observables_dict: dict
        Dictionary with the observables selection
    """
    if (observables_dict['WL']['GCspectro'] or
            observables_dict['GCphot']['GCspectro']):
        observables_dict['WL']['GCspectro'] = False
        observables_dict['GCphot']['GCspectro'] = False
        warnings.warn(
            'Attention: CLOE only computes cross-correlations '
            'for the photometric survey!')
        warnings.warn(
            "Entries ['WL']['GCspectro'] and ['GCphot']['GCspectro'] "
            "are changed to False.")
    try:
        if (observables_dict['CMBlens']['WL'] and
                not observables_dict['WL']['WL']):
            observables_dict['CMBlens']['CMBlens'] = True
            observables_dict['WL']['WL'] = True
            warnings.warn(
                'Attention: CMB cross correlations requires '
                'auto correlations as well')
        if (observables_dict['CMBlens']['GCphot'] and
                not observables_dict['GCphot']['GCphot']):
            observables_dict['CMBlens']['CMBlens'] = True
            observables_dict['GCphot']['GCphot'] = True
            warnings.warn(
                'Attention: CMB cross correlations requires '
                'auto correlations as well')
        # if observables_dict['CMBisw']['GCphot']:
        #     observables_dict['GCphot']['GCphot'] = True

        if (observables_dict['CMBlens']['GCspectro'] or
                observables_dict['CMBisw']['GCspectro']):
            observables_dict['CMBlens']['GCspectro'] = False
            observables_dict['CMBisw']['GCspectro'] = False
            warnings.warn(
                "Entries ['CMBlens']['GCspectro'] and ['CMBisw']['GCspectro'] "
                "are changed to False.")
    except KeyError:
        pass
    return observables_dict


def ell_checker(specifications_dict_prob):
    """
    Multipoles checker.

    Checks if ells selected by the user for the WL, GCphot
    and the cross-correlation beetween photometric
    probes are greater or equal than 0
    and that `ell_max` is greater or equal than `ell_min`.

    Parameters
    ----------
    specifications_dict_prob: dict
        Dictionary with the specifications for one probe
    """
    ells_ranges = \
        np.array(
           [specifications_dict_prob['bins'][bin_i][bin_j]['ell_range'][0] for
            bin_i in specifications_dict_prob['bins'] for
            bin_j in specifications_dict_prob['bins'][bin_i]])
    if np.any(ells_ranges < 0):
        raise ValueError('Error: not all ells in specifications are positive')
    if np.any(np.diff(ells_ranges, axis=1) < 0):
        raise ValueError('Error: not all ells max are greater than ells min')


def observables_selection_specifications_checker(observables_dict,
                                                 specifications_dict):
    """
    Observables selection and specifications checker.

    Merges in a single dictionary the observable selection
    and the specifications loaded in the :obj:`cobaya_interface.py`.

    Parameters
    ----------
    observables_dict: dict
        Dictionary with the observables selection
    specifications_dict: dict
        Dictionary with the observables specifications
    statistics_photo: string
        Statistics used for photo observables (Fourier or Configuration space)

    Returns
    -------
    Merged dictionary: dict
        Dictionary with observables selection and specifications
    """
    # First check observables selection
    checked_observables_dict = observables_selection_checker(observables_dict)
    # Define empty dict for merging
    merged_dict = {'selection': {}, 'specifications': {}}
    merged_dict['selection'] = checked_observables_dict
    # Check each entry of the observables selection and
    # add specifications
    if checked_observables_dict['WL']['WL']:
        merged_dict['specifications']['WL'] = specifications_dict['WL']
        statistics_WL = specifications_dict['WL']['statistics']
        if statistics_WL == 'angular_power_spectrum':
            ell_checker(specifications_dict['WL'][statistics_WL])
    if checked_observables_dict['GCphot']['GCphot']:
        merged_dict['specifications']['GCphot'] = \
            specifications_dict['GCphot']
        statistics_GCphot = specifications_dict['GCphot']['statistics']
        if statistics_GCphot == 'angular_power_spectrum':
            ell_checker(specifications_dict['GCphot'][statistics_GCphot])
    if checked_observables_dict['GCspectro']['GCspectro']:
        merged_dict['specifications']['GCspectro'] = \
            specifications_dict['GCspectro']
    if checked_observables_dict['WL']['GCphot']:
        merged_dict['specifications']['WL-GCphot'] = \
            specifications_dict['WL-GCphot']
        statistics_XC = specifications_dict['WL-GCphot']['statistics']
        if statistics_XC == 'angular_power_spectrum':
            ell_checker(specifications_dict['WL-GCphot'][statistics_XC])
    if checked_observables_dict['CG']['CG']:
        merged_dict['specifications']['CG'] = \
            specifications_dict['CG']
    # At the moment, these quantities below are not computed
    # by CLOE and we are forcing this selection to be False.
    # Therefore, the specifications are not loaded.
    # They are added here for completeness in the future
    if checked_observables_dict['GCphot']['GCspectro']:
        merged_dict['specifications']['GCphot-GCspectro'] = \
            specifications_dict['GCphot-GCspectro']
    if checked_observables_dict['WL']['GCspectro']:
        merged_dict['specifications']['WL-GCspectro'] = \
            specifications_dict['WL-GCspectro']

    # Include the CMB lensing and iSW probes and specifications if requested
    try:
        if checked_observables_dict['CMBlens']['CMBlens']:
            merged_dict['specifications']['CMBlens'] = \
                specifications_dict['CMBlens']
        if checked_observables_dict['CMBlens']['WL']:
            merged_dict['specifications']['CMBlens-WL'] = \
                specifications_dict['CMBlens-WL']
        if checked_observables_dict['CMBlens']['GCphot']:
            merged_dict['specifications']['CMBlens-GCphot'] = \
                specifications_dict['CMBlens-GCphot']
        if checked_observables_dict['CMBisw']['GCphot']:
            merged_dict['specifications']['ISW-GCphot'] = \
                specifications_dict['ISW-GCphot']
    except KeyError:
        # Not using the CMB observables
        checked_observables_dict['CMBlens'] = {
            'CMBlens': False, 'WL': False, 'GCphot': False}
        checked_observables_dict['CMBisw'] = {'GCphot': False}
    return merged_dict
