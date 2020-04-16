# General import
import numpy as np
import matplotlib.pyplot as plt
import likelihood.cosmo

# Import loglike classes
from likelihood.photometric_survey.shear import Shear
from likelihood.spectroscopic_survey.spec import Spec

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
z_min = 0.001
z_max = 2.5
z_samp = 10
z_win = np.linspace(z_min, z_max, z_samp)

# (GCH): value to be evaluated fsigma8
zk = 0.5

# SJ: temporary (should be varied in MCMC)
b_gal = 1.0

# SJ: temporary (needs to be obtained from Cobaya)
sigma_8 = 1.0

# Initialise cosmological parameter dictionary
likelihood.cosmo.cosmology.initialiseparamlist()


def loglike(like_selection=12,
            _theory={"Pk_interpolator": {"z": z_win,
                                         "k_max": k_max,
                                         "extrap_kmax": 2,
                                         "nonlinear": True,
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
    theory_dic = {'H0': _theory.get_param("H0"),
                  'omch2': _theory.get_param('omch2'),
                  'ombh2': _theory.get_param('ombh2'),
                  'mnu': _theory.get_param('mnu'),
                  'comov_dist': _theory.get_comoving_radial_distance(z_win),
                  'H': _theory.get_H(z_win),
                  'Pk_interpolator': _theory.get_Pk_interpolator(),
                  'Pk_delta': None,
                  'fsigma8': None,
                  'zk': zk,
                  'b_gal': b_gal,
                  'sigma_8': sigma_8,
                  }
    theory_dic['Pk_delta'] = (theory_dic['Pk_interpolator']
                              ['delta_tot_delta_tot'])
    theory_dic['fsigma8'] = _theory.get_fsigma8(theory_dic['zk'])

    # (GCH): Careful! In the future, there should be a
    # way to retrieving _derived
    # (GCH): parameters

    # (ACD): As I understand it from Cobaya docs, _theory.get_param
    # only works from within the likelihood function (this one). So the
    # only way to store them in the 'cosmo' module is to pass them from here,
    # like so:

    likelihood.cosmo.cosmology.cosmoparamdict = theory_dic

    # (GCH) Define loglike variable
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
        shear_ins = Shear()
        loglike = shear_ins.loglike()
    elif like_selection.lower() == "spec":
        spec_ins = Spec()
        loglike = spec_ins.loglike()
    elif like_selection.lower() == 'both':
        shear_ins = Shear()
        spec_ins = Spec()
        loglike_shear = shear_ins.loglike()
        loglike_spec = spec_ins.loglike()
        loglike = loglike_shear + loglike_spec
    else:
        raise CobayaInterfaceError(
            r"Choose like selection 'shear' or 'spec' or 'both'")

    # (GCH) loglike=-0.5*chi2
    return loglike
