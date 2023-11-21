import json
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from pa04.pa04 import *

TEST_DATA_CSV = "pa04/.demographic_data_test_2500.csv"


class TestSampling(unittest.TestCase):

    def test_sampling_small_pop_criterion(self):
        test_pop = np.linspace(0, 99, 100)
        self.assertEqual(None, sample_from_population(test_pop, 20))

    def test_if_seed_used(self):
        test_pop = np.linspace(0, 99, 100)
        test_sample1 = sample_from_population(test_pop, 5, random_seed=1)
        test_sample2 = sample_from_population(test_pop, 5, random_seed=1)
        test_sample3 = sample_from_population(test_pop, 5, random_seed=2)

        self.assertTrue((test_sample1 == test_sample2).all())
        self.assertFalse((test_sample2 == test_sample3).all())

    def test_sample_size(self):
        test_pop = np.linspace(0, 99, 100)
        self.assertEqual(5, len(sample_from_population(test_pop, 5)))

    def test_sample_content(self):
        test_pop = np.linspace(0, 99, 100)
        test_sample_size = 5
        test_sample = sample_from_population(test_pop, test_sample_size)
        for i in test_sample:
            self.assertTrue(i in test_pop)


class TestNormal2(unittest.TestCase):

    def test_z_score(self):
        test_mean = 1100
        test_std = 200
        test_value = 1300
        expected_z_value = 1

        self.assertEqual(expected_z_value, z_score(test_mean, test_std, test_value))


    def test_check_unusual(self):
        test_mean = 1100
        test_std = 200
        test_value1 = 1300
        test_value2 = 2000

        self.assertEqual(False, check_unusual(test_mean, test_std, test_value1))
        self.assertEqual(True, check_unusual(test_mean, test_std, test_value2))


class TestInference(unittest.TestCase):

    def test_suc_fail(self):
        n1 = 1000
        p1 = 0.5
        res1 = True

        n2 = 10
        p2 = 0.1
        res2 = False

        self.assertEqual(res1, suc_fail_condition(p1, n1))
        self.assertEqual(res2, suc_fail_condition(p2, n2))

    def test_central_limit_se(self):
        n = 1000
        p = 0.5
        res = 0.0158114
        self.assertAlmostEqual(res, central_limit_se(p, n))

    def test_conf_interval(self):
        n = 1000
        p = 0.5
        z = 1
        lower = 0.5 - 0.0158114
        upper = 0.5 + 0.0158114
        self.assertAlmostEqual(lower, conf_interval_proportion(p, n, z)[0])
        self.assertAlmostEqual(upper, conf_interval_proportion(p, n, z)[1])


class TestProVacStudy(unittest.TestCase):
    sample_size = 200
    nb_samples = 100
    z_value = 1.96
    data_df = pd.read_csv(TEST_DATA_CSV)
    samples, cintervals, pop_prop_in_cinterval = pro_vac_confidence_interval_study(data_df['pro_vac'],
                                                                                   sample_size,
                                                                                   nb_samples,
                                                                                   z_value)

    def test_sample_size(self):
        for s in self.samples:
            # correct length
            self.assertEqual(self.sample_size, len(s))

    def test_sample_individuality(self):
        s_ref = np.array(self.samples[0])
        for s in self.samples[1:]:
            s = np.array(s)
            self.assertFalse((s_ref == s).all())

    def test_number_of_cintervals_covering_popprop(self):
        in_interval_nb = self.pop_prop_in_cinterval.count(True)
        self.assertTrue(90 < in_interval_nb < 99)



if __name__ == '__main__':
    unittest.main()
