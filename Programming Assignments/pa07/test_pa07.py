import unittest
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pa07 import *


class TestScatter(unittest.TestCase):

    rel = lambda x: 5 + 0.9 * x
    np.random.seed(0)
    x_var = np.random.rand(100) * 100
    y_var = np.array(list(map(rel, x_var)))

    scatter_fig = scatter_raw_xy(x_var, y_var)
    ax_plot = scatter_fig.get_axes()[0]

    def test_sp_labels(self):
        xlabel_ref = "Explanatory Variable"
        ylabel_ref = "Response"
        title_ref = "Raw Data Scatter Plot"
        self.assertEqual(xlabel_ref, self.ax_plot.get_xlabel())
        self.assertEqual(ylabel_ref, self.ax_plot.get_ylabel())
        self.assertEqual(title_ref, self.ax_plot.get_title())

    def test_sp_xy(self):

        data_mat = np.array(self.ax_plot.collections[0].get_offsets())

        self.assertTrue((self.x_var == data_mat[:, 0]).all())
        self.assertTrue((self.y_var == data_mat[:, 1]).all())


class TestCorrelation(unittest.TestCase):

    rel = lambda x: 5 + 0.9 * x
    np.random.seed(0)
    x_var = np.random.rand(100) * 100
    y_var = np.array(list(map(rel, x_var)))
    np.random.seed(0)
    y_var_rnd = np.random.normal(0,1,100)

    def test_cor_1(self):
        self.assertAlmostEqual(correlation(self.x_var, self.y_var), 1)

    def test_cor_rnd(self):
        self.assertAlmostEqual(correlation(self.x_var, self.y_var_rnd), -0.13073032423139339)


class TestLSRModel(unittest.TestCase):

    rel = lambda x: 5 + 0.9 * x
    np.random.seed(0)
    x_var = np.random.rand(100) * 100
    y_var = np.array(list(map(rel, x_var)))

    model = least_squares_regression(x_var, y_var)

    def test_b0(self):
        self.assertAlmostEqual(self.model[0], 5)

    def test_b1(self):
        self.assertAlmostEqual(self.model[1], 0.9)


class TestResiduals(unittest.TestCase):

    rel = lambda x: 5 + 0.9 * x
    np.random.seed(0)
    x_var = np.random.rand(100) * 100
    y_var = np.array(list(map(rel, x_var)))

    def test_residuals(self):
        self.assertTrue((residuals(self.x_var, self.y_var, (5, 0.9)) == np.zeros(100)).all())


if __name__ == '__main__':
    unittest.main()