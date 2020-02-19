#General import 
import numpy as np
import matplotlib.pyplot as plt


#Import loglike classes
from shear import Shear
from spec import Spec

#Error classes
class CobayaInterfaceError(Exception):
    """
    Class to define Exception Error
    """
    pass


def loglike(like_selection='both', _theory={"Pk_interpolator": {"z": zwin, "k_max": kmax, "extrap_kmax": 500, "nonlinear": True,"vars_pairs": ([["delta_tot", "delta_tot"]])}, "comoving_radial_distance": {"z": zwin},"H": {"z": zwin, "units": "km/s/Mpc"}}):
        r""" loglike
        
        External likelihood module called by COBAYA

        Parameters
        ----------
        like_selection: string
              Parameter to specify which likelihood to  use:
              'both' - use WL and GC spec
              'shear' - use WL only
              'spec' - use GC spec only
        _theory: dictionary
                 Theory needs required by Boltzmann Solver
                 and used by COBAYA
        _derived

        Returns
        ----------
        loglikes: float
                  must return -0.5*chi2
        """
       # (GCH) Define loglike variable
       loglike = 0.0

       #(GCH) Select with class to work with based on like_selection 
       #(GCH) Within each if-statement, compute loglike
       if like_selection.lower() == "shear":
            shear_ins = Shear(theory=_theory)
            loglike=shear_ins.loglike()
       elif like_selection.lower() == "spec":
            spec_ins = Spec(theory=_theory)
            loglike=spec_ins.loglike()
       elif like_selection.lower() == 'both':
            shear_ins = Shear(theory=_theory)
            spec_ins = Spec(theory=_theory)
            loglike_shear=shear_ins.loglike()
            loglike_spec=spec_ins.loglike()
            loglike=loglikes_shear+loglikes_spec
       else:
            raise CobayaInterfaceError("Choose like_selection
            'shear'|'spec'|'both'")

        # (GCH) loglike=-0.5*chi2
        return loglike



