"""
pgg_spectro

This module contains recipes for the spectroscopic
galaxy x galaxy power spectrum.
"""

from cloe.non_linear.power_spectrum import PowerSpectrum
from cloe.auxiliary.redshift_bins import find_bin, select_spectro_parameters


class Pgg_spectro_model(PowerSpectrum):
    r"""
    Class for computation of spectroscopic galaxy-galaxy power spectrum.
    """

    def __init__(self, cosmo_dic, nonlinear_dic, misc, redshift_bins):
        super(Pgg_spectro_model, self).__init__(cosmo_dic, nonlinear_dic, misc)
        self.zbins = redshift_bins

    def Pgg_spectro_def(self, redshift, wavenumber, mu_rsd):
        r"""Pgg spectro def.

        Computes the redshift-space galaxy-galaxy power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm gg}^{\rm spectro}(z, k) &=\
            [b_{\rm g}^{\rm spectro}(z) + f(z, k)\mu_{\rm k}^2]^2\
            P_{\rm \delta\delta}(z, k)\\

        Note: either :obj:`k_scale` or :obj:`mu_rsd` must be a :obj:`float`
        (e.g. simultaneously setting both of them to :obj:`numpy.ndarray`
        makes the code crash).

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum
        mu_rsd: float or numpy.ndarray
            Cosine of the angle between the pair separation and
            the line of sight

        Returns
        -------
        Spectroscopic galaxy-galaxy power spectrum: float or numpy.ndarray
            Value of galaxy-galaxy power spectrum
            at a given redshift, wavenumber and :math:`\mu_{\rm k}`
            for galaxy clustering spectroscopic
        """
        bias = self.misc.istf_spectro_galbias(redshift)
        growth = self.theory['f_z'](redshift)
        power = self.theory['Pk_delta'].P(redshift, wavenumber)
        pval = (bias + growth * mu_rsd ** 2.0) ** 2.0 * power
        return pval

    def Pgg_spectro_eft(self, redshift, wavenumber, mu_rsd):
        r"""Pgg spectro eft.

        Computes the galaxy-galaxy power spectrum for the spectroscopic probe
        using the eft model.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum
        mu_rsd: float or numpy.ndarray
            Cosine of the angle between wavenumber and the
            line of sight

        Returns
        -------
        EFT spectro galaxy-galaxy power spectrum: float or numpy.ndarray
            Value of the galaxy-galaxy power spectrum at given redshift,
            wavenumber and cosine values for the spectroscopic probe
        """
        zbin = find_bin(redshift, self.zbins, check_bounds=True)
        pval = self.nonlinear_dic['P_kmu'][zbin - 1](wavenumber, mu_rsd,
                                                     grid=False)
        return pval

    def noise_Pgg_spectro(self, redshift, wavenumber, mu_rsd):
        r"""Noise corrections to spectroscopic Pgg.

        Computes the noise contributions to the spectroscopic galaxy power
        spectrum, first selecting the parameters of the corresponding
        spectroscopic bins (through the specified redshift), and then
        evaluating the noise as a function of wavenumber and angle to the
        line of sight.

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum
        mu_rsd: float or numpy.ndarray
            Cosine of the angle between wavenumber and the
            line of sight

        Returns
        -------
        Noise contribution for the spectroscopic galaxy power spectrum at a
        given redshift, wavenumber and angle to the line of sight
        """
        nuisance_dict = self.theory['nuisance_parameters']
        params = select_spectro_parameters(redshift, nuisance_dict, self.zbins)
        aP = params['aP'] if 'aP' in params.keys() else 0.0
        e0k2 = params['e0k2'] if 'e0k2' in params.keys() else 0.0
        e2k2 = params['e2k2'] if 'e2k2' in params.keys() else 0.0
        Psn = params['Psn'] if 'Psn' in params.keys() else 0.0
        noise = Psn * ((1.0 + aP) + (e0k2 + e2k2 * mu_rsd**2) * wavenumber**2)
        return noise

    def Pgdelta_spectro_def(self, redshift, wavenumber, mu_rsd):
        r"""Pgdelta spectro def.

        Computes the redshift-space galaxy-matter power spectrum for the
        spectroscopic probe.

        .. math::
            P_{\rm g \delta}^{\rm spectro}(z, k) &=\
            [b_{\rm g}^{\rm spectro}(z)+f(z, k)\mu_{\rm k}^2]\
            [1+f(z, k)\mu_{\rm k}^2] P_{\rm \delta\delta}(z, k)\\

        Note: either :obj:`k_scale` or :obj:`mu_rsd` must be a :obj:`float`
        (e.g. simultaneously setting both of them to :obj:`numpy.ndarray`
        makes the code crash).

        Parameters
        ----------
        redshift: float
            Redshift at which to evaluate the power spectrum
        wavenumber: float or list or numpy.ndarray
            Wavenumber at which to evaluate the power spectrum
        mu_rsd: float or numpy.ndarray
            Cosine of the angle between the pair separation
            and the line of sight

        Returns
        -------
        Spectroscopic galaxy-matter power spectrum: float or numpy.ndarray
            Value of galaxy-matter power spectrum
            at a given redshift, wavenumber and :math:`\mu_{\rm k}`
            for galaxy clustering spectroscopic
        """
        bias = self.misc.istf_spectro_galbias(redshift)
        growth = self.theory['f_z'](redshift)
        power = self.theory['Pk_delta'].P(redshift, wavenumber)
        pval = ((bias + growth * mu_rsd ** 2.0) *
                (1.0 + growth * mu_rsd ** 2.0)) * power
        return pval
