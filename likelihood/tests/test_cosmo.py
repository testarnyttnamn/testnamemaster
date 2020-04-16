"""UNIT TESTS FOR COSMO

This module contains unit tests for the cosmo module.
=======

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from scipy import integrate
from ..cosmo import *


class cosmoinitTestCase(TestCase):

    def setUp(self):
        cosmology.initialiseparamlist()
        self.H0check = 65.0

    def tearDown(self):
        self.H0check = None

    def test_cosmo_init(self):
        emptflag = bool(cosmology.cosmoparamdict)
        npt.assert_equal(emptflag, False,
                         err_msg='Cosmology dictionary not initialised '
                                 'correctly.')

    def test_cosmo_asign(self):
        cosmology.cosmoparamdict['H0'] = self.H0check
        npt.assert_almost_equal(cosmology.cosmoparamdict['H0'],
                                self.H0check,
                                err_msg='Cosmology dictionary assignment '
                                        'failed')
