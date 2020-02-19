#General imports
import numpy as np

#Import auxilary classes
from estimates import Estimates

class Spec:
    """
    Class for GC spectroscopy observable
    """
    def __init__(self, theory=theory):
        """
        Parameters
        ----------
        theory: dictionary
               Theory needs from COBAYA
        """
        self.theory=theory


    def loglike(self):
        """
        Returns
        -------
        loglike: float
                loglike for GC spectroscopy observable
        """
        loglike=0.0
        return loglike


