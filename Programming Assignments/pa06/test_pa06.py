import json
import math
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pa06.pa06 import *


class TestOSMCI(unittest.TestCase):
    np.random.seed(0)
    data1 = np.random.normal(180, 10, 100)
    data2 = np.random.normal(165, 10, 100)
    data3 = np.random.normal(178, 10, 100)

    def test_ci_values(self):
        ci1 = osm_confidence_interval(self.data1, 0.05)
        self.assertAlmostEqual(ci1[0], 178.58814820996596)
        self.assertAlmostEqual(ci1[1], 182.60801210072376)

    def test_sig_level10(self):
        ci2 = osm_confidence_interval(self.data2, 0.10)
        self.assertAlmostEqual(ci2[0], 164.0935230510729)
        self.assertAlmostEqual(ci2[1], 167.54673636388384)

    def test_height_study(self):
        gender = [1] * 100
        gender.extend([0] * 100)

        height = list(self.data1)
        height.extend(list(self.data2))

        female_mean_height_ci, male_mean_height_ci, overlap = osm_height_study(np.array(gender),
                                                                               np.array(height),
                                                                               0.01)
        self.assertAlmostEqual(female_mean_height_ci[0], 163.08898462335932)
        self.assertAlmostEqual(female_mean_height_ci[1], 168.55127479159742)
        self.assertAlmostEqual(male_mean_height_ci[0], 177.93763709023938)
        self.assertAlmostEqual(male_mean_height_ci[1], 183.25852322045034)
        self.assertFalse(overlap)

    def test_overlap(self):
        gender = [1] * 100
        gender.extend([0] * 100)

        height = list(self.data1)
        height.extend(list(self.data3))

        height2 = list(self.data2)
        height2.extend(list(self.data1))

        study_result1 = osm_height_study(np.array(gender), np.array(height), 0.01)
        study_result2 = osm_height_study(np.array(gender), np.array(height2), 0.01)

        self.assertTrue(study_result1[2])
        self.assertFalse(study_result2[2])

class TestANOVA(unittest.TestCase):
    np.random.seed(0)
    score1 = np.rint(np.random.normal(100, 10, 50))
    score2 = np.rint(np.random.normal(100, 10, 50))
    score3 = np.rint(np.random.normal(100, 10, 50))
    score4 = np.rint(np.random.normal(110, 10, 50))

    problem1 = np.column_stack((score1, score2, score3))
    problem2 = np.column_stack((score1, score2, score4))

    def test_msg(self):
        msg1 = anova_msg(self.problem1)
        msg2 = anova_msg(self.problem2)

        self.assertAlmostEqual(msg1, 92.34000000000023)
        self.assertAlmostEqual(msg2, 1217.220000000002)

    def test_mse(self):
        mse1 = anova_mse(self.problem1)
        mse2 = anova_mse(self.problem2)

        self.assertAlmostEqual(mse1, 105.56585034013605)
        self.assertAlmostEqual(mse2, 103.52734693877551)

    def test_f_statistic(self):
        f1 = anova_f(self.problem1)
        f2 = anova_f(self.problem2)

        self.assertAlmostEqual(f1, 0.8747146894803407)
        self.assertAlmostEqual(f2, 11.757473131421472)

    def test_anova_study(self):
        study1 = anova_student_study(self.problem1, 0.05)
        study1b = anova_student_study(self.problem1, 0.5)
        study2 = anova_student_study(self.problem2, 0.1)

        self.assertAlmostEqual(study1[0], 0.41913983376983854)
        self.assertFalse(study1[1])
        self.assertTrue(study1b[1])

        self.assertAlmostEqual(study2[0], 1.8335123716011505e-05)
        self.assertTrue(study2[1])


if __name__ == '__main__':
    unittest.main()
