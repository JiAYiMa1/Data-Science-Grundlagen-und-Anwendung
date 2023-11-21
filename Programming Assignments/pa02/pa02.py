# Authors
# Jiayi Ma
# Dingyi Zhou
import numpy as np

### YOU ARE NOT ALLOWED TO USE ADDITIONAL MODULES !!! ###

def dsl_mean(values):
    """
    Calculate the sample mean of input values.
    Args:
        values: list of numerical values

    Returns:
        sample mean of input values
    """
    values_mean = 0
    if  values == []:
        return None
    else :values_mean = float(sum(values))/len(values)
    return  values_mean


def dsl_median(values):
    """
    Calculate median of list of numerical values.
    Args:
        values: list of numerical values

    Returns:
        median of input values
    """
    values_median = 0

    values.sort()
    if values == []: return None
    if len(values) % 2 == 0:
        values_median = (values[len(values) // 2] + values[(len(values) - 1) // 2]) / 2
    else:
        values_median = values[len(values) // 2]
    return values_median


def dsl_percentile(values, percent):
    """
    Calculate the percentile given by "percent" of input values.
    Args:
        percent: percent value (int)
        values: list of numerical values

    Returns:
        percentile of input values
    """

    perc = 0
    if values == []:
        return None
    elif len(values):
        list.sort(values)
        a = ((len(values) - 1) * percent / 100)
        c = a - int(a)
        perc = values[int(a)] - c * values[int(a)] + c * values[int(a) + 1]
        return perc






def dsl_mode(values):
    """
    Calculate mode of input values.
    Args:
        values: list of values

    Returns:
        mode of input values
    """

    values_mode = 0
    nums = values
    nums.sort()
    if values == []: return None
    else:
        counts = dict()
        for i in nums:
         counts[i] = counts.get(i, 0) + 1
         values_mode = max(counts, key=counts.get)



    return values_mode


def dsl_range(values):
    """
    Calculate range of input values.
    Args:
        values: list of numerical values

    Returns:
        min, max of input values
    """
    if values == []: return None
    else: min_max = (min(values),max(values))



    return min_max


def dsl_var(values):
    """
    Calculate variance of input values.
    Args:
        values: list of numerical values

    Returns:
        variance of input values
    """
    values_var = 0

    if values == []: return None
    else:
        mean = sum(values) / len(values)
        values_var = sum((i - mean) ** 2 for i in values) / (len(values)-1)


    return values_var


def dsl_std(values):
    """
    Calculate standard deviation of input values.
    Args:
        values: list of numerical values

    Returns:
        std of input values
    """
    values_std = 0

    variance = dsl_var(values)
    if values == []: return None
    else:values_std = variance ** 0.5



    return values_std


if __name__ == '__main__':
    # This part of the code gets executed if it is ran directly with the python interpreter
    # "python pa02.py"
    # You can use this space to try out the functions you implemented above

    # define some test inputs
    test_array = [1,2,3,5,6,4]

    # apply a function and print the result
    # (uncomment the next lines before running to see the result)
    test_array_mean = dsl_mean(test_array)
    test_array_median = dsl_median(test_array)
    test_array_mode = dsl_mode(test_array)
    test_array_variance = dsl_var(test_array)
    test_array_std = dsl_std(test_array)
    test_array_per = dsl_percentile(test_array,75)

    print("Test Aarray: " + str(test_array))
    print("Mean: " + str(test_array_mean))
    print("Median: " + str(test_array_median))
    print("Mode: " + str(test_array_mode))
    print("Min_Max: " + str(dsl_range(test_array)))
    print("Variance: " + str(test_array_variance))
    print("std: " + str(test_array_std))
    print(test_array_per)