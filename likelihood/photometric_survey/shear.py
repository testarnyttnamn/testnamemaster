# General imports
import numpy as np
import likelihood.cosmo
from scipy import integrate

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
        if self.theory['r_z_func'] is None:
            raise Exception('No interpolated function for comoving distance '
                            'exists in cosmo_dic.')

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
        bin_z_max: float
                   Upper limit of bin
        bin_z_min: float
                   Lower limit of bin
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
        n_z_normalized = galdist.n_i
        # SJ: k-indep bias, let us follow the IST:F approach for now
        # W_i_G = self.phot_galbias(k, z) * n_z_normalized(z) * \
        #     self.theory['H'](z)
        W_i_G = self.phot_galbias(bin_z_min, bin_z_max) * n_z_normalized(z) * \
            self.theory['H'](z)
        return W_i_G

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

    def loglike(self):
        """
        Returns loglike for Shear observable
        """

        # likelihood value is currently only a place holder!
        loglike = 0.0
        return loglike
