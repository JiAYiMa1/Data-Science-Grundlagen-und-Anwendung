"""
linear regression
"""
#Author: Jiayi Ma
#Co-Author: Dingyi Zhou
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import math
import sqlite3




def scatter_raw_xy(x_array, y_array):
    """
    Create scatter plot for raw data
    Args:
        x_array: x data
        y_array: y data

    Returns:
        matplotlib.pyplot figure object containing the plot
    """
    fig1 = plt.figure(1)
    plt.scatter(x_array, y_array, alpha=0.2)
    plt.xlabel('Explanatory Variable')
    plt.ylabel('Response')
    plt.title('Raw Data Scatter Plot')
    plt.grid()


    return fig1


def correlation(x_array, y_array):
    """
    Calculate correlation.
    Args:
        x_array: x data
        y_array: y data

    Returns:
        correlation between x and y
    """
    mean_x = np.mean(x_array)
    mean_y = np.mean(y_array)
    std_x = np.std(x_array,ddof=1)
    std_y = np.std(y_array,ddof=1)
    n = len(x_array)
    sum = 0
    for i in range(0,n):
        sum += (x_array[i]-mean_x)*(y_array[i]-mean_y)/(std_x * std_y)
    correlation = sum/(n-1)
    return correlation



def least_squares_regression(x_array, y_array):
    """
    Estimate model parameters b0 and b1 using least squares regression
    Args:
        x_array: x data
        y_array: y data

    Returns:
        tuple with b0, b1

    """
    r = correlation(x_array,y_array)
    std_x_array = np.std(x_array,ddof=1)
    std_y_array = np.std(y_array,ddof=1)
    y_mean = np.mean(y_array)
    x_mean = np.mean(x_array)
    b_1 = r*(std_y_array/std_x_array)
    b_0 = y_mean - x_mean * b_1
    return (b_0,b_1)


def residuals(x_array, y_array, model):
    """
    Calculate residuals for given data and model
    Args:
        x_array: x data
        y_array: y data
        model: (b0, b1) tuple containing model parameters

    Returns:
        residuals (numpy array)
    """
    y_model = model[1] * x_array + model[0]
    residuals = y_array - y_model
    return residuals


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(description='Read a filepath.')
    parser.add_argument('filepath', type=str, help='Path to file containing data.')
    args = parser.parse_args()

    # read the data
    con = sqlite3.connect(args.filepath)
    cursor = con.cursor()
    sql = "SELECT * FROM lin_reg"
    values = cursor.execute(sql)
    lin_reg = pd.DataFrame(data=values)
    lin_reg = lin_reg.rename(columns={'x': 'y'})
    print(lin_reg)
    x_array = lin_reg[1].to_numpy().reshape(-1,1)
    y_array = lin_reg[2].to_numpy().reshape(-1,1)

    # draw the plot and save in pdf
    fig1 = scatter_raw_xy(x_array, y_array)
    fig1.savefig('/Users/jiayima/dsl-ws21-ma/pa07.pdf')

    # calculate correlation
    R = correlation(x_array, y_array)
    print("R = %s"%(R))

    # plot the linear model with least squares regression approach and save in pdf
    model = least_squares_regression(x_array,y_array)
    y_model = model[1] * x_array + model[0]
    print("y = %s * x + %s"%(model[1],model[0]))
    fig2 = plt.figure(2)
    plt.scatter(x_array,y_array)
    plt.plot(x_array, y_model, color='r')
    plt.xlabel('Explanatory Variable')
    plt.ylabel('Response')
    plt.title('Raw Data Scatter Plot')
    plt.grid()
    plt.show()
    fig2.savefig('/Users/jiayima/Desktop/Linearity.pdf')

    # plot histogram of residuals and save in pdf
    residuals = residuals(x_array,y_array,model)
    fig3 = plt.figure(3)
    plt.hist(residuals, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("A histogram of the residuals")
    plt.show()
    fig3.savefig('/Users/jiayima/Desktop/Nearly normal residuals.pdf')

    # plot residuals against explanatory variable, Boxplot and save in pdf
    residuals_sum = np.mean(residuals)
    residuals_array = []
    for i in range(0,len(x_array)):
        residuals_array.append(residuals_sum)
    fig5, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].scatter(x_array, residuals)
    ax[0].plot(x_array,residuals_array,linewidth=1,linestyle='--',color='g')
    ax[0].set_xlabel('Explanatory Variable')
    ax[0].set_ylabel('Residuals')
    ax[0].set_title('Test Constant variability')
    ax[1].boxplot(residuals)
    ax[1].set_xlabel('Explanatory Variable')
    ax[1].set_ylabel('Residuals')
    ax[1].set_title('Test Constant variability Box Plot')
    fig5.tight_layout()
    plt.show()
    fig5.savefig('/Users/jiayima/Desktop/Constant variability.pdf')


    pass