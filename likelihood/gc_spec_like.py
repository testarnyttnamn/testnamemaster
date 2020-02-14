# General imports
import numpy as np


#Sampling in redshift
zmin  = 0.001
zmax  = 2.5
zsamp = 100
zwin  = np.linspace(zmin,zmax,zsamp)

#Specifications for Power Spectra
kmax = 25.0

def gc_spec_like(_theory={"Pk_interpolator": {"z": zwin, "k_max": kmax, "extrap_kmax": 500, "nonlinear": True,"vars_pairs": ([["delta_tot", "delta_tot"]])}, "comoving_radial_distance": {"z": zwin},"H": {"z": zwin, "units": "km/s/Mpc"}}):

    #How to call cosmological quantities (example from Matteo)
    H0      = _theory.get_param("H0")
    Omega_m = (_theory.get_param('omch2')+_theory.get_param('ombh2')+(_theory.get_param('mnu')*(3.046/3)**0.75)/94.0708)/(H0/100)**2. 
    H   = _theory.get_H(zwin)
    Pk_interpolator = _theory.get_Pk_interpolator()
    PKdelta = Pk_interpolator["delta_tot_delta_tot"]
    
    return 0




