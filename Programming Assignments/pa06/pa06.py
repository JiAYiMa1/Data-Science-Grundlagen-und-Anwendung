# Authors
# Jiayi Ma
# Dingyi Zhou

# imports
import argparse
import math
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats


def osm_confidence_interval(data, sig):
    """
    Compute confidence interval for one sample mean.
    Args:
        data: data points
        sig: desired significance level

    Returns:
        tuple defining the confidence interval
    """
    conf_interval = 0
    mean = np.mean(data)
    se = stats.sem(data)
    df = len(data) - 1
    t_df = stats.t.ppf(sig / 2, df=df)
    min = mean + se * t_df
    max = mean - se * t_df
    conf_interval = (min, max)

    return conf_interval


def osm_height_study(gender, height, sig):
    """
    Height study for gender groups
    Args:
        gender: numpy array containing gender information
        height: numpy array containing height information
        sig: desired level of significance

    Returns:
        female_mean_height_ci,
        male_mean_height_ci,
        overlap

    """
    female_mean_height_ci = 0
    male_mean_height_ci = 0
    overlap = 0
    male_height = height[gender == 1]
    female_mean_height = height[gender == 0]
    male_mean_height_ci = osm_confidence_interval(male_height, sig)
    female_mean_height_ci = osm_confidence_interval(female_mean_height, sig)

    if max(female_mean_height_ci) < min(male_mean_height_ci):
        overlap = False
    elif min(female_mean_height_ci) > max(male_mean_height_ci):
        overlap = False
    else:
        overlap = True

    return female_mean_height_ci, male_mean_height_ci, overlap


def anova_msg(problem_array):
    """
    Calculate the MSG
    Args:
        problem_array: 2D numpy array describing a multi group problem

    Returns:
        MSG value
    """
    msg = 0
    k = len(problem_array[0])
    df_g = k - 1
    mean_all = np.mean(problem_array)
    ssg = 0
    for i in range(0, k):
        problem_array_group = problem_array[:, i]
        mean_group = np.mean(problem_array_group)
        ssg += len(problem_array_group) * ((mean_group - mean_all) ** 2)
    msg = ssg / df_g

    return msg


def anova_mse(problem_array):
    """
    Calculate the MSE
    Args:
        problem_array: 2D numpy array describing a multi group problem

    Returns:
        MSE value
    """
    mse = 0
    sse = 0
    df = pd.DataFrame(problem_array)
    for col, colitems in df.iteritems():
        col_mean = colitems.mean()
        sse += ((colitems - col_mean) ** 2).sum()
    mse = sse / (df.size - len(df.columns))
    return mse


def anova_f(problem_array):
    """
    Calculate the F statistic
    Args:
        problem_array: 2D numpy array describing a multi group problem

    Returns:
        F value
    """
    f = 0
    msg = anova_msg(problem_array)
    mse = anova_mse(problem_array)
    f = msg / mse

    return f


def anova_student_study(problem_array, sig):
    """
    ANOVA for student scores
    Args:
        problem_array: 2D numpy array describing a multi group problem
        sig: desired level of significance

    Returns:
        p_value: P value for F test
        reject_H_0: bool indicating whether H_0 is to be rejected
    """
    p_value = 0
    reject_H_0 = 0
    df = pd.DataFrame(problem_array)
    df_e = df.size - len(df.columns)
    df_g = len(df.columns) - 1
    f = anova_f(problem_array)
    p_value = stats.f.sf(f, df_g, df_e)
    if p_value > sig:
        reject_H_0 = False
    else:
        reject_H_0 = True

    return p_value, reject_H_0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()

    np.random.seed(0)
    data1 = np.random.normal(180, 10, 100)
    ci1 = osm_confidence_interval(data1, 0.05)
    score1 = np.rint(np.random.normal(100, 10, 50))
    score2 = np.rint(np.random.normal(100, 10, 50))
    score3 = np.rint(np.random.normal(100, 10, 50))
    score4 = np.rint(np.random.normal(110, 10, 50))

    problem1 = np.column_stack((score1, score2, score3))
    problem2 = np.column_stack((score1, score2, score4))
    print(anova_msg(problem1))
    print(anova_mse(problem1))
    con = sqlite3.connect(args.db)
    cursor = con.cursor()
    sql = "SELECT * FROM persons"
    values = cursor.execute(sql)
    persons = pd.DataFrame(data=values, )
    print(persons)
    gender_array = persons[1].to_numpy()
    height_array = persons[3].to_numpy()
    result = osm_height_study(gender_array,height_array,0.05)
    print(result)#((164.81397222115794, 165.11947511230562), (185.02375873451783, 185.23811820653012), False)

    sql_1 = "SELECT * FROM scores"
    values_1 = cursor.execute(sql_1)
    scores = pd.DataFrame(data=values_1, )
    cursor.close()
    print(scores)
    scores.to_numpy()

    score_class_1 = scores[1].to_numpy()
    score_class_2 = scores[2].to_numpy()
    score_class_3 = scores[3].to_numpy()
    score_class = np.column_stack((score_class_1,score_class_2,score_class_3))
    result_score = anova_student_study(score_class,0.05)
    print(result_score)#for significance level = 0.05 result_score = (2.2716649262633596e-05, True)
    """ study questions:
    1) Task 2 study: a) Results: ((164.81397222115794, 165.11947511230562), (185.02375873451783, 185.23811820653012), False) for desired level of significance = 0.05
                     b) Conclusion: we are 95% confident that the average female height is between 164.8 cm and 165.1 cm, the average male height is between 185.0 cm and 185.2 cm. The two calculated confidence intervals (for female and male) do not overlap.
    2) Task 5 study: a) Results: (2.2716649262633596e-05, True) for significance level = 0.05
                     b) Conclusion: we can reject Ho, because p-Value < 0.05, that is, the data provide strong evidence that the average scores of the exact same test varies by different classes.  
"""