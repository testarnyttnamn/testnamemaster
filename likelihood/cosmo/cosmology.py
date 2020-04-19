# -*- coding: utf-8 -*-
"""Cosmology

Class to store cosmological parameters and functions.
"""

import numpy as np

class CosmologyError(Exception):
    r"""
    Class to define Exception Error
    """
    pass

class Cosmology:
"""
Class for GC spectroscopy observable
"""

    def __init__(self, cobaya_dic):
        """
        Parameters
        ----------
        cobaya_dic: dictionary
                    cosmological quantities from cobaya
        """
        self.cosmo_dic = cobaya_dic


    def growth_factor(self, zs, ks):
        """
        Calculates growth_factor according as
        
        .. math::
                   D(z) &=\sqrt{P(z)/P(z=0)}\\

        Parameters
        ----------
        zs: array
            list of redshift for the power spectrum
        ks: array
            list of modes for the power spectrum

        Returns
        -------
        growth factor

        """
        # GCH: Careful! This should be updated in the future!
        # we want to obtain delta directly from Cobaya
        # (in process)
        # This quantity depends on z and k
        D_z = self.cosmo_dic['PKdelta'].P(zs, ks)
        D_z = np.sqrt(D_z / self.cosmo_dic['PKdelta'].P(0, ks))
        return D_z

    def update_cosmo_dic(self, zs, ks):
        """
        Update the dictionary with other cosmological quantities
        
        Parameters
        ----------
        zs: array
            list of redshift for the power spectrum
        ks: array
            list of modes for the power spectrum
            
        Returns
        -------
        growth factor
        
        """
        # (GCH): this function is superfluous
        # just in case we want to have always
        # an updated dictionary with D_z and f
        try:
            self.cosmo_dic['D_z'] = self.growth_factor(zs, ks)
        raise:
            CosmologyError(
            r"Computation error in D_z")
        

