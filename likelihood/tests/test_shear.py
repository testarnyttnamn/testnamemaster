"""UNIT TESTS FOR SHEAR

This module contains unit tests for the Shear sub-module of the
photometric survey module.
=======

"""

# (GCH): Use Cobaya Model wrapper

from cobaya.model import get_model
from unittest import TestCase
import numpy as np
import numpy.testing as npt
from scipy import integrate
from scipy import interpolate
from ..cosmo.cosmology import Cosmology
from likelihood.cobaya_interface import loglike
from ..photometric_survey import shear


class cosmoinitTestCase(TestCase):

    def setUp(self):
        self.H0 = 67.0
        self.c = 3.0e5
        self.omch2 = 0.12
        self.ombh2 = 0.022
        self.rfn = interpolate.InterpolatedUnivariateSpline(
            np.linspace(0.0, 2.6, 20), np.linspace(0.0, 2.6, 20), ext=2)
        self.flatnz = interpolate.InterpolatedUnivariateSpline(
            np.linspace(0.0, 2.6, 20), np.ones(20), ext=2)
        self.sheartest = shear.Shear({'H0': self.H0,
                                      'c': self.c,
                                      'omch2': self.omch2,
                                      'ombh2': self.ombh2,
                                      'r_z_func': self.rfn})

        self.integrand_check = -1.0
        self.wbincheck = 9.143694637057992e-09

    def tearDown(self):
        self.integrand_check = None
        self.wbincheck = None

    def test_w_integrand(self):
        int_comp = self.sheartest.w_gamma_integrand(0.1, 0.2, self.flatnz)
        npt.assert_almost_equal(int_comp, self.integrand_check,
                                err_msg='WL kernel integrand check failed.')

    def test_w_bin(self):
        int_comp = self.sheartest.w_kernel_gamma(0.1, self.flatnz, 2.5)
        npt.assert_almost_equal(int_comp, self.wbincheck,
                                err_msg='WL kernel check failed.')

    def test_rzfunc_exception(self):
        npt.assert_raises(Exception, shear.Shear,
                          {'H0': self.H0,
                           'c': self.c,
                           'omch2': self.omch2,
                           'ombh2': self.ombh2})
