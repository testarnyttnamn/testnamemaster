# General imports
import numpy as np
import likelihood.cosmo

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
