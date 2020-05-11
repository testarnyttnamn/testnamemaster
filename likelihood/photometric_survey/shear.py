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
        Calculates integrand of interal in WL lensing kernel.

        .. math::
        \int_{z}^{z_{\rm max}}
        {{\rm d}z^{\prime} n_{i}^{\rm WL}(z^{\prime}) \left [ 1 -
        \frac{\tilde{r}(z)}{\tilde{r}(z^{\prime}} \right ]}

        Parameters
        ----------
        zprime: float
            redshift parameter that will be integrated over.
        z: float
            Redshift at which kernel is being evaluated.
        nz: function
            Galaxy distribution function for the tomographic bin for which the
            kernel is currently being evaluated.
        Returns
        -------
        Integrand value
        """

        wint = nz(zprime) * (1.0 - (self.theory['r_z_func'](z) /
                                    self.theory['r_z_func'](zprime)))
        return wint

    def w_kernel_gamma(self, z, tomo_bin, z_max):
        """
        Calculates the W^{\gamma} lensing kernel for a given tomographic bin.

        .. math::
        W_{i}^{\gamma}(z) = \frac{3 H_0}{2 c}
        \Omega_{{\rm m},0} (1 + z) \Sigma(z,k) \tilde{r}(z)
        \int_{z}^{z_{\rm max}}
        {{\rm d}z^{\prime} n_{i}^{\rm WL}(z^{\prime}) \left [ 1 -
        \frac{\tilde{r}(z)}{\tilde{r}(z^{\prime}} \right ]}

        Parameters
        ----------
        z: float
            Redshift at which kernel is being evaluated.
        tomo_bin: function
            Galaxy distribution function for the tomographic bin for
            which the kernel is currently being evaluated.
        z_max: float
            Maximum redshift of survey, up to which lensing kernel will be
            evaluated.
        Returns
        -------
        Value of lensing kernel for specified bin at specified redshift.
        """
        H0 = self.theory['H0']
        c = self.theory['c']
        O_m = ((self.theory['omch2'] / (H0 / 100.0)**2.0) +
               (self.theory['ombh2'] / (H0 / 100.0)**2.0))
        # (ACD): Note that impact of MG is currently neglected (\Sigma=1).
        W_val = (1.5 * (H0 / c) * O_m * (1.0 + z) * (
                self.theory['r_z_func'](z) /
                (c / H0)) * integrate.quad(self.w_gamma_integrand, a=z,
                                           b=z_max, args=(z, tomo_bin))[0])
        return W_val
