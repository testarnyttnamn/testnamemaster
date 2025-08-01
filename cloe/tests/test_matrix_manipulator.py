"""UNIT TESTS FOR MATRIX MANIPULATOR

This module contains unit tests for the
:obj:`matrix_manipulator` module.

"""


from unittest import TestCase
from unittest.mock import patch
import cloe.auxiliary.matrix_manipulator as matrix_manipulator
import numpy


class MatrixManipulator_test(TestCase):

    def setUp(self):
        self.nRows1, self.nCols1, self.nRows2, self.nCols2 =\
            numpy.random.randint(low=10, high=50, size=4)
        self.mat1 = numpy.random.randn(self.nRows1, self.nCols1)
        self.mat2 = numpy.random.randn(self.nRows2, self.nCols2)
        self.desiredMatrix = numpy.block([
            [self.mat1, numpy.zeros((self.nRows1, self.nCols2))],
            [numpy.zeros((self.nRows2, self.nCols1)), self.mat2]
        ])
        self.rtol = 1e-4

    def tearDown(self):
        pass

    # test that the size and content of the output matrix is as expected
    def test_merge_matrices(self):
        outMatrix = matrix_manipulator.merge_matrices(self.mat1, self.mat2)
        self.assertEqual(outMatrix.shape[0], self.desiredMatrix.shape[0],
                         f'Unexpected number of rows in output matrix')
        self.assertEqual(outMatrix.shape[1], self.desiredMatrix.shape[1],
                         f'Unexpected number of columns in output matrix')
        numpy.testing.assert_allclose(
            actual=outMatrix,
            desired=self.desiredMatrix,
            rtol=self.rtol,
            err_msg=f'Unexpected value in output matrix'
        )
