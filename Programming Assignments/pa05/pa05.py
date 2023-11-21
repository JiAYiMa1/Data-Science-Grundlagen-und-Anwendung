# Authors
# Jiayi Ma
# Dingyi Zhou

import argparse
import math
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import norm
from scipy.stats import chisquare


def mc_create_z_lookup(confidence_levels, nb_samples, random_seed):
    """
    Create a z level lookup for the specified percent values using Monte Carlo Simulation
    Args:
        confidence_levels: confidence levels
        nb_samples: number of draws in simulation
        random_seed: seed for initializing RNG

    Returns: list of z values corresponding to percent_values input

    """

    np.random.seed(random_seed)
    values = np.random.normal(0, 1, nb_samples)

    z_values = []
    for i in range(len(confidence_levels)):
        d = np.percentile(values, (100 - confidence_levels[i]) / 2, axis=0)
        z_values.append(abs(d))
    return z_values


def chi2_dsl(observed_cnts, expected_cnts):
    """
    Calculate chi-square statistic for given pair of observed- and expected counts.
    Args:
        observed_cnts: observed counts
        expected_cnts: expected counts

    Returns: chi-square value
    """
    chi_square = 0
    for i in range(len(expected_cnts)):
        if expected_cnts[i] < 5:
            return None
        else:

            chi_square += ((observed_cnts[i] - expected_cnts[i]) ** 2) / expected_cnts[i]
    return chi_square


def chi2_dice_study(dice_results, significance_level):
    """
    Use Chi2 test to determine whether results are from a unfair dice or not
    Args:
        dice_results: results from throwing a dice (numpy array)
        significance_level: desired level of significance

    Returns:
        unfair: bool indicating unfairness
        p_value: calculated p-value
    """
    unfair = False
    p_value = 0
    ex = []
    ob = []

    for i in range(1, 7):
        x = np.count_nonzero(dice_results == i)
        ob.append(x)
        c = len(dice_results) / 6
        ex.append(c)
    chi_square = chi2_dsl(ob, ex)
    p_value = 1 - stats.chi2.cdf(chi_square, 5)

    if p_value < significance_level:
        unfair = True
    else:
        unfair = False

    return unfair, p_value


def two_way_table_dice_study(dice_1, dice_2, significance_level):
    """
    Test independence in two-way table
    Args:
        dice_1: results from first dice
        dice_2: results from second dice
        significance_level: desired significance level

    Returns:
        dependence: bool indicating dependence
        p_value: calculated p-value

    """
    zahl = max(np.max(dice_1), np.max(dice_2))
    ob_1 = np.zeros(zahl)
    ob_2 = np.zeros(zahl)
    ex_1 = []
    ex_2 = []
    dependence = False
    p_value = 0
    for i in range(0, len(ob_1)):
        ob_1[i] = np.count_nonzero(dice_1 == i + 1)
    for j in range(0, len(ob_2)):
        ob_2[j] = np.count_nonzero(dice_2 == j + 1)
    total = ob_2 + ob_1
    for i in range(0, len(total)):
        a = total[i] * len(dice_1) / (len(dice_1) + len(dice_2))
        ex_1.append(a)
    for i in range(0, len(total)):
        b = total[i] * len(dice_2) / (len(dice_1) + len(dice_2))
        ex_2.append(b)
    chi_square_1 = chi2_dsl(ob_1, ex_1)
    chi_square_2 = chi2_dsl(ob_2, ex_2)
    chi_square = chi_square_1 + chi_square_2
    p_value = 1 - stats.chi2.cdf(chi_square, 5)

    if p_value < significance_level:
        dependence = True
    else:
        dependence = False
    return dependence, p_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()
    print(mc_create_z_lookup([90, 95, 99], 10000, 1))
    dice_1 = np.random.randint(1, 7, size=1000)
    dice_2 = np.random.randint(1, 6, size=1000)
    print(chi2_dice_study(dice_1, 0.05))
    print(two_way_table_dice_study(dice_1, dice_2, 0.05))




    
