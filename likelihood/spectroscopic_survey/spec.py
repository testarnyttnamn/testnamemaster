"""
module: spec

Prototype computation of spectroscopic galaxy clustering likelihood.
"""

# Global
import numpy as np

# Import auxilary classes
# (GCH): for instance, estimates from general_specss


class Spec:
    """
    Class for GC spectroscopy observable
    """

    def __init__(self, theory=None):
        """
        Parameters
        ----------
        theory: dictionary
               Theory needs from COBAYA
        """
        self.theory = theory

    def beta_eqs(self):
        """
        Computation of Eqns 25 - 27 of Euclid IST:L documentation,
        found below:
        https://www.overleaf.com/read/pvfkzvvkymbj
        """

        # SJ: Set up power spectrum [note weirdly Cobaya has it as P(z,k)
        # SJ: instead of the more common P(k,z)]
        Pk_interpolator = self.theory['Pk_interpolator']
        Pk_delta = self.theory['Pk_delta']
        # SJ: For now, fix redshift (and scale),
        # fix galaxy bias = 1, and fix sigma8(z) = 1
        # Cobaya does not seem to allow for either growth
        # rate alone or sigma8 alone
        # to be called yet (only their combination).
        b_gal = 1.0
        sigma8 = 1.0
        # compute Eqns 25-27
        beta = self.theory['fsigma8'] / sigma8 / b_gal
        Pk_gal = (b_gal**2.0) * \
            self.theory['Pk_delta'].P(self.theory['zk'], 0.02)
        P0k = (1.0 + 2.0 / 3.0 * beta + 1.0 / 5.0 * beta**2.0) * Pk_gal
        P2k = (4.0 / 3.0 * beta + 4.0 / 7.0 * beta**2.0) * Pk_gal
        P4k = (8.0 / 35.0 * beta**2.0) * Pk_gal

        # (GCH): maybe save as attributes P0k, P2k, P4k?

    def loglike(self):
        """
        Returns
        -------
        loglike: float
                loglike for GC spectroscopy observable
                ln(likelihood) = -1/2 chi**2
        """
        # SJ: This will be the log-likelihood;
        # for now just return P(z,k) for fixed z and k.
        self.beta_eqs()
        return self.theory['Pk_delta'].P(0.5, 0.02)
