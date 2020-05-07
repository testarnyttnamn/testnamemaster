# General imports
import numpy as np
import likelihood.cosmo

# Import auxilary classes
from ..general_specs.estimates import Galdist

# General error class


class ShearError(Exception):
    r"""
    Class to define Exception Error
    """
    pass


class Shear:
    """
    Class for Shear observable
    """

    def __init__(self, cosmo_dic):
        """
        Constructor of the class Shear

        Parameters
        ----------
        cosmo_dic: dictionary
           cosmological dictionary from cosmo
        """

        self.theory = cosmo_dic

    def bias(self, k, z):
        # (GCH): This should be implemented by Shahab
        b = 1
        return b

    def GC_window(self, bin_i, bin_j, bin_z_min, bin_z_max, k, z):
        """
        Implements GC window

        Parameters
        ----------
        bin_i: list, float
           Redshift bounds of bin i (lower, higher)
        bin_j: list, float
           Redshift bounds of bin j (lower, higher)
        k: float
           Scale at which to evaluate the bias
        z: float
           Redshift at which to evaluate distribution.

        Returns
        -------
        W_i_G: float
           GCph window function

        Notes
        -----
        .. math::
            W_i^G(k, z) &=b(k, z)n_i(z)/\bar{n_i}H(z)\\
        """

        # (GCH): create instance from Galdist class
        try:
            galdist = Galdist(bin_i, bin_j)
        except ShearError:
            print('Error in initializing the class Galdist')
        # (GCH): call n_z_normalized from Galdist
        n_z_normalized = galdist.p_up(bin_z_min, bin_z_max)
        W_i_G = self.bias(k, z) * n_z_normalized(z) * self.theory['H'](z)
        return W_i_G

    def loglike(self):
        """
        Returns loglike for Shear observable
        """

        # likelihood value is currently only a place holder!
        loglike = 0.0
        return loglike
