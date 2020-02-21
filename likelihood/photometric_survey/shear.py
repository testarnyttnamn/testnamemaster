# General imports
import numpy as np

# Import auxilary classes
from ..general_specs.estimates import Galdist


class Shear:
    """
    Class for Shear observable
    """

    def __init__(self, theory=None):
        """
        Parameters
        ----------
        """
        self.theory = theory

    def loglike(self):
        """
        Returns loglike for Shear observable
        """
        loglike = 0.0
        return loglike
