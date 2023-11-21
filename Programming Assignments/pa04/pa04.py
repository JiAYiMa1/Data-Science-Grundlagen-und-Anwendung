# Authors
# Jiayi Ma
# Dingyi Zhou

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def sample_from_population(population, sample_size, random_seed=0):
    """
    Create a sample from a given poulation.
    Args:
        population: Array of elements defining a population
        sample_size: Size of sample to return
        random_seed: seed for the RNG (default 0)

    Returns:
        Array with sample from population
    """
    sample = 0
    np.random.seed(random_seed)

    if (sample_size/np.size(population)) > 0.1:
        return None

    else:

        sample = np.random.choice(population, size = sample_size, replace = False)
        return sample


def z_score(inp_mean, inp_std, value):
    """
    Calc Z-score
    Args:
        inp_mean: mean
        inp_std: std
        value: value to calc z-score for

    Returns: z-score for value

    """
    z_score = 0

    z_score = (value-inp_mean)/inp_std

    return z_score


def check_unusual(inp_mean, inp_std, value):
    """
    Check if a value is considered unusual in the context of the given distribution
    Args:
        inp_mean: mean
        inp_std: std
        value: value to check

    Returns: bool

    """
    unusual = True

    if (value - inp_mean) <= 2 * inp_std:
         unusual = False

    return unusual


def suc_fail_condition(p, n):
    """
    Success failure condition
    Args:
        p: proportion
        n: sample size

    Returns: bool

    """


    check = False
    if n*p >= 10 and n*(1-p)>=10:
        check = True

    return check


def central_limit_se(p, n):
    """
    Standard error according to central limit theorem
    Args:
        p: proportion
        n: sample size

    Returns: standard error

    """

    if suc_fail_condition(p,n) == False:
        return None
    clt_se = 0
    clt_se = (p) / math.sqrt(n)

    return clt_se


def conf_interval_proportion(p, n, z):
    """
    Confidence interval for a proportion
    Args:
        p: proportion
        n: sample size
        z: z-value

    Returns: tuple (lower_bound, upper_bound)

    """


    lower = 0
    upper = 0
    lower = p - z * math.sqrt((p*(1-p)/n))
    upper = p + z * math.sqrt((p*(1-p)/n))

    return lower, upper


def pro_vac_confidence_interval_study(population, sample_size, nb_samples, z_value):
    """
    This function takes multiple samples of fixed size from a population and calculates the confidence intervals for
    the contained sample proportions.
    Args:
        population: data with elements [0,1]
        sample_size: desired size of samples
        nb_runs: number of samples (confidence intervals) to create
        z_value: z value corresponding to the confidence level (e.g. 1.96 for 95%)

    Returns:
        list of numpy arrays representing the samples,
        list of tuples representing the confidence intervals (lower bound, upper bound) for the sample proportions,
        list of bools representing whether an interval covers the true population proportion

    """

    samples = []
    confidence_intervals = []
    pop_proportion_in_cinterval = []
    pop_pro = population.sum() / population.size

    for i in range(nb_samples):
        samples.append(sample_from_population(population,sample_size,random_seed=i))
        proportion = sum(samples[i])/sample_size
        confidence_intervals.append(conf_interval_proportion(proportion,sample_size,z_value))
        if pop_pro > min(confidence_intervals[i]) and pop_pro < max(confidence_intervals[i]):
            pop_proportion_in_cinterval.append(True)

        else:
            pop_proportion_in_cinterval.append(False)
    a = sum(pop_proportion_in_cinterval)/len(pop_proportion_in_cinterval)


    return samples, confidence_intervals, pop_proportion_in_cinterval



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, required=True)
    args = parser.parse_args()
    #df_raw  = pd.read_csv(args.datafile)
    #print(pro_vac_confidence_interval_study(df_raw['pro_vac'],1000,100,1.96))



