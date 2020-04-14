# General imports
import numpy as np
import likelihood.cosmo

# Import auxilary classes
from ..general_specs.estimates import Galdist


class Shear:
    """
    Class for Shear observable
    """

    def __init__(self):
        """
        Parameters
        ----------
        """
        self.theory = likelihood.cosmo.cosmology.cosmoparamdict

    def loglike(self):
        """
        Returns loglike for Shear observable
        """

        # likelihood value is currently only a place holder!
        loglike = 0.0
        return loglike
