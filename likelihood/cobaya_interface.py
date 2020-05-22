# General import
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

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


# SJ: For now, example sampling in wavenumber (k)
k_min = 0.002
k_max = 0.2
k_samp = 100
k_win = np.linspace(k_min, k_max, k_samp)

# SJ: For now, example sampling in redshift (z)
z_min = 0.0
z_max = 2.5
z_samp = 10
z_win = np.linspace(z_min, z_max, z_samp)

# (GCH): value to be evaluated fsigma8
zk = 0.5

# SJ: temporary (should be varied in MCMC)
b_gal = 1.0

# SJ: temporary (needs to be obtained from Cobaya)
sigma_8 = 1.0


def loglike(like_selection=12,
            _theory={"Pk_interpolator": {"z": z_win,
                                         "k_max": k_max,
                                         "extrap_kmax": 2,
                                         "nonlinear": False,
                                         "vars_pairs": ([["delta_tot",
                                                          "delta_tot"]])},
                     "comoving_radial_distance": {"z": z_win},
                     "H": {"z": z_win, "units": "km/s/Mpc"},
                     "fsigma8": {"z": z_win, "units": None}}):
    r""" loglike

    External likelihood module called by COBAYA

    Parameters
    ----------
    like_selection: int
              Parameter to specify which likelihood to  use:
              12 - use WL and GC spec (default value)
              1 - use WL only
              2 - use GC spec only
              this will updated in the future by strings
    _theory: dictionary
            Theory needs required by Boltzmann Solver
            and used by COBAYA
    _derived: dictionary
            Future dictionary where new quantities
            calculated by the likelihood can be saved
            and passed to the sampler

    Returns
    ----------
    loglikes: float
            must return -0.5*chi2
    """
    # (GCH): re-define _theory needs of COBAYA
    # (GCH): I guess that this can be made automatically asking
    # (GCH): the theory code which parameters it has... search for it!!
    # (GCH): for the moment, this is
    # (GCH): Careful! In the future, there should be a
    # way to retrieving _derived
    # (GCH): parameters

    # (ACD): As I understand it from Cobaya docs, _theory.get_param
    # only works from within the likelihood function (this one). So the
    # only way to store them in the 'cosmo' module is to pass them from here,
    # like so:
    # (GCH): Updated from ACD. cosmology is a class now
    # Cosmology will calculate growth factor and rate too

    cosmo = Cosmology()
    try:
        cosmo.cosmo_dic['H0'] = _theory.get_param("H0")
        cosmo.cosmo_dic['omch2'] = _theory.get_param('omch2')
        cosmo.cosmo_dic['ombh2'] = _theory.get_param('ombh2')
        cosmo.cosmo_dic['mnu'] = _theory.get_param('mnu')
        cosmo.cosmo_dic['comov_dist'] = \
            _theory.get_comoving_radial_distance(z_win)
        cosmo.cosmo_dic['H'] = UnivariateSpline(z_win, _theory.get_H(z_win))
        cosmo.cosmo_dic['Pk_interpolator'] = _theory.get_Pk_interpolator()
        cosmo.cosmo_dic['zk'] = zk
        cosmo.cosmo_dic['b_gal'] = b_gal
        cosmo.cosmo_dic['sigma_8'] = sigma_8
        cosmo.cosmo_dic['Pk_delta'] = (
            cosmo.cosmo_dic['Pk_interpolator']['delta_tot_delta_tot'])
        cosmo.cosmo_dic['fsigma8'] = _theory.get_fsigma8(
            cosmo.cosmo_dic['zk'])
        cosmo.cosmo_dic['z_win'] = z_win
        cosmo.interp_comoving_dist()
    except CobayaInterfaceError:
        print('Cobaya theory needs could not be pass to cosmo module')

    # (GCH): loglike computation
    loglike = 0.0
    # (GCH): issue with cobaya to pass strings to external likelihood
    # as parameter
    if like_selection == 1:
        like_selection = "shear"
    elif like_selection == 2:
        like_selection = "spec"
    elif like_selection == 12:
        like_selection = "both"

    # (GCH) Select with class to work with based on like_selection
    # (GCH) Within each if-statement, compute loglike
    if like_selection.lower() == "shear":
        shear_ins = Shear(cosmo.cosmo_dic)
        loglike = shear_ins.loglike()
    elif like_selection.lower() == "spec":
        spec_ins = Spec(cosmo.cosmo_dic)
        loglike = spec_ins.loglike()
    elif like_selection.lower() == 'both':
        shear_ins = Shear(cosmo.cosmo_dic)
        spec_ins = Spec(cosmo.cosmo_dic)
        loglike_shear = shear_ins.loglike()
        loglike_spec = spec_ins.loglike()
        loglike = loglike_shear + loglike_spec
    else:
        raise CobayaInterfaceError(
            r"Choose like selection 'shear' or 'spec' or 'both'")

    # (GCH) loglike=-0.5*chi2
    return loglike
