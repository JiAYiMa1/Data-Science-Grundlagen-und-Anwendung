import unittest
import json

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

from pa03.pa03 import *

TEST_DATA_CSV = "pa03/.demographic_data_test.csv"
TEST_DATA_JSON = "pa03/.data_test.json"


class TestReadIn(unittest.TestCase):
    dict_json = json.load(open(TEST_DATA_JSON, 'r'))
    dict_data_ref = {
        'id': np.array(dict_json['id']),
        'gender': np.array(dict_json['gender']),
        'age': np.array(dict_json['age']),
        'height': np.array(dict_json['height']),
        'pro_vac': np.array(dict_json['pro_vac'])
    }
    dict_data_read = read_demographic_data(TEST_DATA_CSV)

    def test_ri_keys(self):
        self.assertEqual(self.dict_data_ref.keys(), self.dict_data_read.keys())

    def test_ri_arrays(self):
        for k in self.dict_data_ref.keys():
            self.assertTrue((self.dict_data_ref[k] == self.dict_data_read[k]).all())


class TestAgeHeightScatter(unittest.TestCase):
    dict_json = json.load(open(TEST_DATA_JSON, 'r'))
    dict_data_ref = {
        'id': np.array(dict_json['id']),
        'gender': np.array(dict_json['gender']),
        'age': np.array(dict_json['age']),
        'height': np.array(dict_json['height']),
        'pro_vac': np.array(dict_json['pro_vac'])
    }
    ah_scatter_fig = age_height_scatter_plot(dict_data_ref['age'], dict_data_ref['height'])
    ax_plot = ah_scatter_fig.get_axes()[0]

    def test_sp_labels(self):
        xlabel_ref = "Age in years"
        ylabel_ref = "Height in cm"
        title_ref = "Age-Height Scatter Plot"
        self.assertEqual(xlabel_ref, self.ax_plot.get_xlabel())
        self.assertEqual(ylabel_ref, self.ax_plot.get_ylabel())
        self.assertEqual(title_ref, self.ax_plot.get_title())

    def test_sp_color(self):
        color_ref = '#c477e3'
        self.assertEqual(color_ref, matplotlib.colors.to_hex(self.ax_plot.collections[0].get_facecolors()[0]))


    def test_sp_ticks(self):
        xtick_ref = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ytick_ref = [120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220]

        self.assertTrue((xtick_ref == self.ax_plot.get_xticks()).all())
        self.assertTrue((ytick_ref == self.ax_plot.get_yticks()).all())

    def test_sp_xy(self):

        data_mat = np.array(self.ax_plot.collections[0].get_offsets())

        self.assertTrue((self.dict_data_ref['age'] == data_mat[:, 0]).all())
        self.assertTrue((self.dict_data_ref['height'] == data_mat[:, 1]).all())


class TestGenderHeightBoxplot(unittest.TestCase):
    dict_json = json.load(open(TEST_DATA_JSON, 'r'))
    dict_data_ref = {
        'id': np.array(dict_json['id']),
        'gender': np.array(dict_json['gender']),
        'age': np.array(dict_json['age']),
        'height': np.array(dict_json['height']),
        'pro_vac': np.array(dict_json['pro_vac'])
    }

    x = 237
    gender = [0] * x
    gender.extend([1] * x)
    gender = np.array(gender)
    height_male = list(np.linspace(1, x, x))
    height_female = list(np.linspace(1, x, x))
    height = []
    height.extend(height_female)
    height.extend(height_male)
    height = np.array(height)

    gh_boxplot_fig, gh_boxplot_dict = gender_height_boxplot(gender, height)
    ax_plot = gh_boxplot_fig.get_axes()[0]

    def test_bp_median(self):
        female_median = np.median(self.height_female)
        male_median = np.median(self.height_male)

        self.assertEqual(female_median, self.gh_boxplot_dict["medians"][0].get_ydata()[0])
        self.assertEqual(male_median, self.gh_boxplot_dict["medians"][1].get_ydata()[0])

    def test_bp_quartiles(self):
        female_quartiles = [np.percentile(self.height_female, 25), np.percentile(self.height_female, 75)]
        male_quartiles = [np.percentile(self.height_male, 25), np.percentile(self.height_male, 75)]

        self.assertEqual(female_quartiles, [min(self.gh_boxplot_dict["boxes"][0].get_ydata()),
                                            max(self.gh_boxplot_dict["boxes"][0].get_ydata())])
        self.assertEqual(male_quartiles, [min(self.gh_boxplot_dict["boxes"][1].get_ydata()),
                                            max(self.gh_boxplot_dict["boxes"][1].get_ydata())])

    def test_bp_whiskers(self):
        female_whiskers = [np.percentile(self.height_female, 2.5, interpolation="higher"), np.percentile(self.height_female, 97.5, interpolation="lower")]
        male_whiskers = [np.percentile(self.height_male, 2.5, interpolation="higher"), np.percentile(self.height_male, 97.5, interpolation="lower")]

        self.assertEqual(female_whiskers, [min(self.gh_boxplot_dict["whiskers"][0].get_ydata()),
                                            max(self.gh_boxplot_dict["whiskers"][1].get_ydata())])
        self.assertEqual(male_whiskers, [min(self.gh_boxplot_dict["whiskers"][2].get_ydata()),
                                          max(self.gh_boxplot_dict["whiskers"][3].get_ydata())])

    def test_bp_labels(self):
        ylabel_ref = "Height in cm"
        title_ref = "Gender-Height Box Plot"
        xlabel_ref = ["female", "male"]

        self.assertEqual(ylabel_ref, self.ax_plot.get_ylabel())
        self.assertEqual(title_ref, self.ax_plot.get_title())
        self.assertEqual(xlabel_ref, self.ax_plot.xaxis.major.formatter.seq)


class TestFreedmanDiaconis(unittest.TestCase):
    dict_json = json.load(open(TEST_DATA_JSON, 'r'))
    dict_data_ref = {
        'id': np.array(dict_json['id']),
        'gender': np.array(dict_json['gender']),
        'age': np.array(dict_json['age']),
        'height': np.array(dict_json['height']),
        'pro_vac': np.array(dict_json['pro_vac'])
    }

    def test_fd_binwidth(self):
        self.assertEqual(np.lib.histograms._hist_bin_fd(self.dict_data_ref["height"], 0),
                         freedman_diaconis_bin_width(self.dict_data_ref["height"]))


class TestHeightHistogram(unittest.TestCase):
    dict_json = json.load(open(TEST_DATA_JSON, 'r'))
    dict_data_ref = {
        'id': np.array(dict_json['id']),
        'gender': np.array(dict_json['gender']),
        'age': np.array(dict_json['age']),
        'height': np.array(dict_json['height']),
        'pro_vac': np.array(dict_json['pro_vac'])
    }

    height_hist_fig, n, bins, patches = height_histogram(dict_data_ref["height"])
    n_ref, bins_ref, patches_ref = plt.hist(dict_data_ref["height"], bins="fd")

    ax_plot = height_hist_fig.get_axes()[0]

    def test_hist_bins(self):
        self.assertTrue((self.bins_ref == self.bins).all())

    def test_hist_values(self):
        self.assertTrue((self.n_ref == self.n).all())

    def test_hist_labels(self):
        xlabel_ref = 'Height in cm'
        ylabel_ref = 'Number of Cases'
        title_ref = 'Height Histogram'

        self.assertEqual(xlabel_ref, self.ax_plot.get_xlabel())
        self.assertEqual(ylabel_ref, self.ax_plot.get_ylabel())
        self.assertEqual(title_ref, self.ax_plot.get_title())


class TestPDFSampling(unittest.TestCase):
    test_sampling_points = np.linspace(-3, 3, 61)

    def test_norm_pdf_sampling(self):
        ref = norm.pdf(self.test_sampling_points, loc=0, scale=1)
        sol = sample_norm_pdf(0,1,self.test_sampling_points)
        for i in range(len(self.test_sampling_points)):
            self.assertAlmostEqual(ref[i], sol[i])


class TestNormPDFPlot(unittest.TestCase):
    test_sampling_points = np.linspace(-3, 3, 61)

    pdf_line_fig = normal_pdf_line_plot(0,1,test_sampling_points)
    ax_plot = pdf_line_fig.get_axes()[0]

    def test_lp_labels(self):
        ylabel_ref = 'PDF'
        title_ref = 'Normal PDF Line Plot'
        self.assertEqual(ylabel_ref, self.ax_plot.get_ylabel())
        self.assertEqual(title_ref, self.ax_plot.get_title())

    def test_lp_color(self):
        color_ref = '#000000'
        self.assertEqual(color_ref, matplotlib.colors.to_hex(self.ax_plot.lines[0].get_color()))

    def test_lp_lims(self):
        xlim_ref = (min(self.test_sampling_points), max(self.test_sampling_points))
        self.assertEqual(xlim_ref, self.ax_plot.get_xlim())

    def test_lp_xy(self):

        data_mat = np.array(self.ax_plot.lines[0].get_data())

        self.assertTrue((self.test_sampling_points == data_mat[0]).all())

        ref = norm.pdf(self.test_sampling_points, loc=0, scale=1)
        for i in range(len(self.test_sampling_points)):
            self.assertAlmostEqual(ref[i], data_mat[1][i])


if __name__ == '__main__':
    unittest.main()
