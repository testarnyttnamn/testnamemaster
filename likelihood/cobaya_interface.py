# General import
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Cobaya import of general Likelihood class
from cobaya.likelihood import Likelihood

# Import likelihoods classes
from likelihood.photometric_survey.shear import Shear
from likelihood.spectroscopic_survey.spec import Spec
from likelihood.cosmo.cosmology import Cosmology

# Error classes


class CobayaInterfaceError(Exception):
    r"""
    Class to define Exception Error
    """
    pass


class EuclidLikelihood(Likelihood):
    aliases = ["myOld"]
    # Speed in evaluations/second (after theory inputs calculated).
    speed = 500
    params = {"like_selection": None}

    def initialize(self):
        # SJ: For now, example sampling in wavenumber (k)
        self.k_min = 0.002
        self.k_max = 0.2
        self.k_samp = 100
        self.k_win = np.linspace(self.k_min, self.k_max, self.k_samp)

        # SJ: For now, example sampling in redshift (z)
        self.z_min = 0.0
        self.z_max = 2.5
        self.z_samp = 10
        self.z_win = np.linspace(self.z_min, self.z_max, self.z_samp)

        # (GCH): value to be evaluated fsigma8
        self.zk = 0.5

        # SJ: temporary (should be varied in MCMC)
        self.b_gal = 1.0

        # SJ: temporary (needs to be obtained from Cobaya)
        self.sigma_8 = 1.0

    def get_requirements(self):
        return {"Pk_interpolator": {"z": self.z_win,
                                    "k_max": self.k_max,
                                    "nonlinear": False,
                                    "vars_pairs": ([["delta_tot",
                                                     "delta_tot"]])},
                "comoving_radial_distance": {"z": self.z_win},
                "Hubble": {"z": self.z_win, "units": "km/s/Mpc"},
                "fsigma8": {"z": self.z_win, "units": None}}

    def logp(self, **params_values):
        r""" logp

        External likelihood module called by COBAYA

        Parameters
        ----------
        like_selection: int
              Parameter to specify which likelihood to  use:
              12 - use WL and GC spec (default value)
              1 - use WL only
              2 - use GC spec only
              this will updated in the future by strings
        Returns
        ----------
        loglikes: float
            must return -0.5*chi2
        """

        cosmo = Cosmology()
        try:
            cosmo.cosmo_dic['H0'] = self.provider.get_param("H0")
            cosmo.cosmo_dic['omch2'] = self.provider.get_param('omch2')
            cosmo.cosmo_dic['ombh2'] = self.provider.get_param('ombh2')
            cosmo.cosmo_dic['mnu'] = self.provider.get_param('mnu')
            cosmo.cosmo_dic['comov_dist'] = \
                self.provider.get_comoving_radial_distance(self.z_win)
            cosmo.cosmo_dic['H'] = UnivariateSpline(
                self.z_win, self.provider.get_Hubble(self.z_win))
            cosmo.cosmo_dic['Pk_interpolator'] = \
                self.provider.get_Pk_interpolator(nonlinear=False)
            cosmo.cosmo_dic['zk'] = self.zk
            cosmo.cosmo_dic['b_gal'] = self.b_gal
            cosmo.cosmo_dic['sigma_8'] = self.sigma_8
            cosmo.cosmo_dic['Pk_delta'] = \
                self.provider.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=False)
            cosmo.cosmo_dic['fsigma8'] = self.provider.get_fsigma8(
                cosmo.cosmo_dic['zk'])
            cosmo.cosmo_dic['z_win'] = self.z_win
        except CobayaInterfaceError:
            print('Cobaya theory requirements \
                  could not be pass to cosmo module')

        return self.log_likelihood(cosmo.cosmo_dic, **params_values)

    def log_likelihood(self, dictionary, **data_params):

        # (GCH): loglike computation
        loglike = 0.0
        # (GCH): issue with cobaya to pass strings to external likelihood
        # as parameter
        like_selection = data_params['like_selection']
        if like_selection == 1:
            like_selection = "shear"
        elif like_selection == 2:
            like_selection = "spec"
        elif like_selection == 12:
            like_selection = "both"

        # (GCH) Select with class to work with based on like_selection
        # (GCH) Within each if-statement, compute loglike
        if like_selection.lower() == "shear":
            shear_ins = Shear(dictionary)
            loglike = shear_ins.loglike()
        elif like_selection.lower() == "spec":
            spec_ins = Spec(dictionary)
            loglike = spec_ins.loglike()
        elif like_selection.lower() == 'both':
            shear_ins = Shear(dictionary)
            spec_ins = Spec(dictionary)
            loglike_shear = shear_ins.loglike()
            loglike_spec = spec_ins.loglike()
            loglike = loglike_shear + loglike_spec
        else:
            raise CobayaInterfaceError(
                r"Choose like selection 'shear' or 'spec' or 'both'")

        # (GCH) loglike=-0.5*chi2
        return loglike
