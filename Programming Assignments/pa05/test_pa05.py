import json
import math
import unittest
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import norm
from scipy.stats import chisquare

from pa05.pa05 import *



class TestMCZLookUp(unittest.TestCase):

    def test_if_seed_used(self):
        test_z_value1 = mc_create_z_lookup([95], 100, random_seed=1)
        test_z_value2 = mc_create_z_lookup([95], 100, random_seed=1)
        test_z_value3 = mc_create_z_lookup([95], 100, random_seed=2)

        self.assertTrue(test_z_value1 == test_z_value2)
        self.assertFalse(test_z_value1 == test_z_value3)

    def test_z_values(self):
        z_values_10k = mc_create_z_lookup([90,95,99], 10000, random_seed=0)
        z_90 = 1.645
        z_95 = 1.960
        z_99 = 2.576

        self.assertAlmostEqual(z_90, z_values_10k[0], 1)
        self.assertAlmostEqual(z_95, z_values_10k[1], 1)
        self.assertAlmostEqual(z_99, z_values_10k[2], 1)


class TestChi2(unittest.TestCase):
    
    np.random.seed(0)
    uniform_dist1000_exp = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    uniform_dist40_exp = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    obs_1000 = np.random.randint(10, size=1000)
    tmp, obs_dist_1000 = np.unique(obs_1000, return_counts=True)

    obs_40 = np.random.randint(10, size=40)
    tmp, obs_dist_40 = np.unique(obs_1000, return_counts=True)


    def test_expected_counts(self):
        self.assertTrue(chi2_dsl(self.obs_dist_1000, self.uniform_dist1000_exp) is not None)
        self.assertTrue(chi2_dsl(self.obs_dist_40, self.uniform_dist40_exp) is None)


    def test_chi2_values(self):
        self.assertAlmostEqual(chisquare(self.obs_dist_1000, self.uniform_dist1000_exp)[0], chi2_dsl(self.obs_dist_1000, self.uniform_dist1000_exp))


class TestChi2DiceStudy(unittest.TestCase):

    def test_fair_dice(self):
        np.random.seed(0)
        dice_1 = np.random.randint(1, 7, size=1000)
        p_ref1 = 0.67841206943743
        unfairness, p_val = chi2_dice_study(dice_1, 0.05)

        self.assertFalse(unfairness)
        self.assertAlmostEqual(p_ref1, p_val)

    def test_unfair_dice(self):
        np.random.seed(0)
        dice_2 = np.random.randint(1, 6, size=1000)
        p_ref2 = 4.8789466040336383 * math.exp(-42)
        unfairness, p_val = chi2_dice_study(dice_2, 0.05)

        self.assertTrue(unfairness)
        self.assertAlmostEqual(p_ref2, p_val)


class TestTwoWayTableStudy(unittest.TestCase):

        def test_not_dependent(self):
            np.random.seed(0)
            dice_1 = np.random.randint(1, 7, size=1000)
            dep_bool, p = two_way_table_dice_study(dice_1, dice_1, 0.05)

            self.assertFalse(dep_bool)
            self.assertAlmostEqual(1, p)

        def test_dependent(self):
            np.random.seed(0)
            dice_1 = np.random.randint(1, 7, size=1000)
            np.random.seed(0)
            dice_2 = np.random.randint(1, 6, size=1000)
            dep_bool, p = two_way_table_dice_study(dice_1, dice_2, 0.05)
            p_ref = 1.4900542256273283 * math.exp(-38)
            self.assertTrue(dep_bool)
            self.assertAlmostEqual(p_ref, p)


if __name__ == '__main__':
    unittest.main()
