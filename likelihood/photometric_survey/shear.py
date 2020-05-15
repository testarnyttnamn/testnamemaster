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

    # SJ: k-indep bias for now
    # def bias(self, k, z):
    def phot_galbias(self, bin_z_min, bin_z_max):
        """
        Returns the photometric galaxy bias.
        For now use Eqn 133 in arXiv:1910.09273

        Parameters
        ----------
        # SJ: k-indep bias for now
        # k: float
        #    Scale at which to evaluate the bias
        # z: float
        #    Redshift at which to evaluate distribution.
        bin_z_max: float
                   Upper limit of bin
        bin_z_min: float
                   Lower limit of bin

        Returns
        -------
        b: float
           galaxy bias

        Notes
        -----
        .. math::
            b(z) &= \sqrt{1+\bar{z}}\\
        """

        # SJ: We could eventually have a bias parameter for each
        # SJ: tomographic bin (also see yaml file)
        # phot_galbias = [params_values.get(p, None) for p in \
        #     ['phot_b1', 'phot_b2', 'phot_b3', 'phot_b4', 'phot_b5', \
        #     'phot_b6', 'phot_b7', 'phot_b8', 'phot_b9', 'phot_b10']]

        b = np.sqrt(1.0 + (bin_z_min + bin_z_max) / 2.0)
        # SJ: Yet another option, not used
        # b = np.sqrt(1 + z)
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
        # SJ: k-indep bias, let us follow the IST:F approach for now
        # W_i_G = self.phot_galbias(k, z) * n_z_normalized(z) * \
        #     self.theory['H'](z)
        W_i_G = self.phot_galbias(bin_z_min, bin_z_max) * n_z_normalized(z) * \
            self.theory['H'](z)
        return W_i_G

    def loglike(self):
        """
        Returns loglike for Shear observable
        """

        # likelihood value is currently only a place holder!
        loglike = 0.0
        return loglike
