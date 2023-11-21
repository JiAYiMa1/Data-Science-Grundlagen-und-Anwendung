# Authors
# YOUR FULL NAME：Jiayi Ma
# YOUR PARTNER'S FULL NAME:Dingyi Zhou

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def read_demographic_data(filepath):
    """
    Read in a file with the same form as demographic_data.csv
    Args:
        filepath: Path to the file to read in

    Returns:
        Dictionary with column labels as keys and numpy arrays as dictionary values.
    """

    data_file = open(filepath, "r")
    lines = data_file.readlines()

    ID_array = np.empty(100, dtype=object)
    age_array = np.zeros(100)
    gender_array = np.zeros(100)
    height_array = np.zeros(100)
    pro_vac_array = np.zeros(100)
    counter = 0
    for line in lines[1:]:
        line_work = line.strip('\n').split(",")

        ID_array[counter] = int(line_work[0], 16)
        gender_array[counter] = float(line_work[1])
        age_array[counter] = float(line_work[2])
        height_array[counter] = float(line_work[3])
        pro_vac_array[counter] = float(line_work[4])
        counter += 1

    demographic_dict = {'id': ID_array, 'gender': gender_array,
                        'age': age_array, 'height': height_array,
                        'pro_vac': pro_vac_array}

    # Hint: Dont forget to pay attention to the type of the values before adding them

    return demographic_dict


def age_height_scatter_plot(age_array, height_array):
    """
    Create a scatter plot with 'age in years' on the x-axis and 'height in cm' on the y-axis.
    Args:
        age_array: numpy array containing age values
        height_array: numpy array containing height values

    Returns:
        matplotlib.pyplot figure object containing the plot
    """
    age_height_figure = plt.figure()
    plt.xlabel("Age in years")
    plt.ylabel("Height in cm")
    plt.title("Age-Height Scatter Plot")
    plt.xlim((0, 100))
    plt.ylim((120, 220))
    my_x_ticks = np.arange(0, 110, 10)
    my_y_ticks = np.arange(120, 230, 10)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plot_age_array = np.array([age_array[i] for i in range(100)])
    plot_height_array = np.array([height_array[i] for i in range(100)])
    plt.scatter(plot_age_array, plot_height_array, color='#c477e3', marker="d")
    plt.grid()
    plt.show()
    return age_height_figure


def gender_height_boxplot(gender_array, height_array):
    """
    Create a figure with separate box plots for height values for females and males
    Args:
        gender_array: numpy array containing gender values [0,1]
        height_array: numpy array containing height values

    Returns:
        matplotlib.pyplot figure object containing the plot
    """
    male_height = height_array[gender_array == 1]
    female_height = height_array[gender_array == 0]

    gender_height_figure = plt.figure()
    bp_dict = plt.boxplot([female_height, male_height], labels=['female', 'male'], whis=(0.45))
    plt.title("Gender-Height Box Plot")
    plt.ylabel("Height in cm")
    plt.show()

    return gender_height_figure, bp_dict


"""    numer_female = 0
    numer_male = 0
    for i in range(len(gender_array)):
        if gender_array[i] == 0:
            numer_female = numer_female + 1
        elif gender_array[i] == 1:
            numer_male = numer_male + 1


    male_heights_array = np.zeros(numer_male)
    femal_heights_array = np.zeros(numer_female)
    f = 0
    m = 0
    for i in range(len(height_array)):
        if gender_array[i] == 1:
            male_heights_array[m] =np.array(height_array[i])
            m = m+1
        elif gender_array[i] == 0:
            femal_heights_array[f] = np.array(height_array[i])
            f = f+1


    male_heights_array.sort()
    femal_heights_array.sort()


    gender_height_figure = plt.figure()
    plt.title('Gender-Height Box Plot')
    plt.ylabel('Height in cm')



    bp_dict= plt.boxplot([femal_heights_array,male_heights_array],labels=["female","male"],whis=(2.5,97.5),
            patch_artist=True
            showmeans=True
            showcaps=True
            boxprops = {'color':'black','facecolor':'#9999ff'}
            showfliers=True , # 设置异常值属性，点的形状、填充色和边框色
            meanprops = {'marker':'D','markerfacecolor':'indianred'}
            medianprops = {'linestyle':'--','color':'orange'})

    plt.show()    """


def normal_pdf_line_plot(inp_mean, inp_std, sampling_points):
    """
    Create a line plot of the normal PDF specified by mean and std for the area covered by sampling points
    Args:
        inp_mean: desired mean
        inp_std: desired std
        sampling_points: points where to sample the pdf

    Returns:
        matplotlib.pyplot figure object containing the plot
    """

    mean = inp_mean
    std = inp_std
    norm_value_array = []

    for i in range(0, len(sampling_points)):
        x = sampling_points[i]
        norm_value = 1 / (math.sqrt(2 * math.pi) * std) * math.exp(-((x - mean) ** 2) / (2 * std ** 2))
        norm_value_array.append(norm_value)
    normal_pdf_figure = plt.figure()
    plt.plot(sampling_points, norm_value_array, color='black', marker='+')
    plt.ylabel("PDF")
    plt.xlim((min(sampling_points), max(sampling_points)))
    plt.title("Normal PDF Line Plot")

    plt.show()
    return normal_pdf_figure


def sample_norm_pdf(inp_mean, inp_std, sampling_points):
    """
    Sample the pdf of a normal disttribution with mean: inp_mean and std: inp_std at the points in sampling_points
    Args:
        inp_mean: desired mean
        inp_std: desired std
        sampling_points: points where to sample the pdf

    Returns: array with values of pdf at points in sampling_points

    """
    mean = inp_mean
    std = inp_std
    norm_value_array = []

    for i in range(0, len(sampling_points)):
        x = sampling_points[i]
        norm_value = 1 / (math.sqrt(2 * math.pi) * std) * math.exp(-((x - mean) ** 2) / (2 * std ** 2))
        norm_value_array.append(norm_value)

    return norm_value_array


def freedman_diaconis_bin_width(data_points):
    """
    Freedman Diaconis bin width calculation
    Args:
        data_points: array containing data

    Returns: width of bins for histogram

    """
    data_points.sort()
    IQR = np.percentile(data_points, (75)) - np.percentile(data_points, (25))
    rd_bin_width = 2 * IQR / ((len(data_points)) ** (1 / 3))
    print("bins")
    print(rd_bin_width)

    return rd_bin_width


def height_histogram(height_array):
    """
    Create a figure with with a height histogram
    Args:
        height_array: numpy array containing height values

    Returns:
        matplotlib.pyplot figure object containing the plot
    """
    rd_bins_width = freedman_diaconis_bin_width(height_array)
    numer = int((max(height_array) - min(height_array)) / rd_bins_width)

    bins = np.arange(min(height_array), max(height_array) + rd_bins_width, rd_bins_width)
    height_figure = plt.figure()

    n, bins, patches = plt.hist(height_array, bins="fd")
    plt.xlabel('Height in cm')
    plt.ylabel('Number of Cases')
    plt.title('Height Histogram')
    plt.show()
    return height_figure, n, bins, patches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, required=True)
    args = parser.parse_args()
    data_dict = read_demographic_data(args.datafile)
    age_array = data_dict['age']
    height_array = data_dict['height']

    age_height_scatter_plot(data_dict['age'], data_dict['height'])

    gender_height_boxplot(data_dict['gender'], data_dict['height'])
    freedman_diaconis_bin_width(data_dict['height'])
    x = height_histogram(data_dict['height'])
    mean_heights = np.mean(data_dict['height'])
    std_heights = np.std(data_dict['height'])
    sp = sample_norm_pdf(mean_heights, std_heights, data_dict['height'])
    normal_pdf_line_plot(mean_heights, std_heights, data_dict['height'])