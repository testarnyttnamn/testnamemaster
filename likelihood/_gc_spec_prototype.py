"""
module: _gc_spec_prototype

Prototype computation of spectroscopic galaxy clustering likelihood.
"""

# Global
import numpy as np

#Local
#from cobaya.likelihoods._base_classes import _DataSetLikelihood

#For now, example sampling in wavenumber (k)
kmin  = 0.002
kmax  = 0.2
ksamp = 100
kwin  = np.linspace(kmin,kmax,ksamp)

#For now, example sampling in redshift (z)
zmin  = 0.001
zmax  = 2.5
zsamp = 10
zwin  = np.linspace(zmin,zmax,zsamp)

#Want to follow the Cobaya approach, for now ignore class structure
#class _gc_spec_prototype(_DataSetLikelihood):
"""
Class for GC-spec likelihood computation.
"""

    #preliminary
    #def initialize_gcspec(self):
    #    self.covmat = 0

#likelihood example
def gc_spec_like(_theory={"Pk_interpolator": {"z": zwin, "k_max": kmax, "extrap_kmax": 2, "nonlinear": True,"vars_pairs": ([["delta_tot", "delta_tot"]])}, "comoving_radial_distance": {"z": zwin},"H": {"z": zwin, "units": "km/s/Mpc"},"fsigma8": {"z": zwin, "units": None}}):
    """
    Computation of likelihood. See Eqns 25 - 27 of Euclid IST:L documentation, found below:

    https://www.overleaf.com/read/pvfkzvvkymbj

    Parameters
    ----------
    z: float array
        Redshift
    k: float array
        Wavenumber (scale)

    Returns
    -------
    float
        ln(likelihood) = -1/2 chi**2
    """

    #Set up power spectrum [note weirdly Cobaya has it as P(z,k) instead of the more common P(k,z)]
    Pk_interpolator = _theory.get_Pk_interpolator()
    PKdelta = Pk_interpolator["delta_tot_delta_tot"]
    
    #For now, fix redshift (and scale), fix galaxy bias = 1, and fix sigma8(z) = 1
    #Cobaya does not seem to allow for either growth rate alone or sigma8 alone to be called yet (only their combination). 
    zk = 0.5
    bias = 1.0
    sigma8 = 1.0

    #compute Eqns 25-27    
    beta = _theory.get_fsigma8(zk)/sigma8/bias
    P0k = (1.0 + 2.0/3.0*beta + 1.0/5.0*beta**2.0)*PKdelta.P(zk,0.02)
    P2k = (4.0/3.0*beta + 4.0/7.0*beta**2.0)*PKdelta.P(zk,0.02)
    P4k = (8.0/35.0*beta**2.0)*PKdelta.P(zk,0.02)
 
    #This will be the log-likelihood; for now just return P(z,k) for fixed z and k.
    return PKdelta.P(zk, 0.02)
