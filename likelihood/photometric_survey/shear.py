# General imports
import numpy as np
import likelihood.cosmo
from scipy import integrate

# Import auxilary classes
from ..general_specs.estimates import Galdist


class Shear:
    """
    Class for Shear observable
    """

    def __init__(self, cosmo_dic):
        """
        Parameters
        ----------
        """
        self.theory = cosmo_dic

    def loglike(self):
        """
        Returns loglike for Shear observable
        """

        # likelihood value is currently only a place holder!
        loglike = 0.0
        return loglike

    def w_gamma_integrand(self, zprime, z, nz):
        """
        Calculates integrand of interal in WL lensing kernel. See Eq. 44 of
        IST:L documentation.

        Parameters
        ----------
        zprime: float
            redshift parameter that will be integrated over.
        z: array
            Redshift at which kernel is being evaluated.
        nz: galaxy distribution function for the tomographic bin for which the
            kernel is currently being evaluated.
        Returns
        -------
        Integrand value
        """

        wint = nz(zprime) * (1.0 - (self.theory['r_z_func'](z)/
                                    self.theory['r_z_func'](zprime)))
        return wint

    def w_kernel_gamma(self, z, tomo_bin):
        """
        Calculates the W^{\gamma} lensing kerneal for a given tomographic bin.
        See Eq. 44 of IST:L documentation.

        Parameters
        ----------
        z: array
            Redshift at which kernel is being evaluated.
        tomo_bin: galaxy distribution function for the tomographic bin for
                  which the kernel is currently being evaluated.
        Returns
        -------
        Value of lensing kernel for specified bin at specified redshift.
        """
        H0 = self.theory['H0']
        c = self.theory['c']
        O_m = ((self.theory['omch2']/(H0/100.0)**2.0) +
               (self.theory['omch2']/(H0/100.0)**2.0))
        # (ACD): Note that impact of MG is currently neglected (\Sigma=1).
        W_val = (1.5 * (H0/c) * O_m * (1.0 + z) * (self.theory['r_z_func'](z)/
                (c/H0)) * integrate.quad(self.w_gamma_integrand, a=0.0,
                                         b=2.5, args=(z, tomo_bin)))[0]
        return W_val
