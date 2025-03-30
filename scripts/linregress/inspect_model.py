# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300

import pickle
import sys

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import linregress
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
import seaborn as sns
import matplotlib.pyplot as plt

def check_model_residual(model_file):

    # Load the model from the pickle file
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    residuals = model.resid_response #GLMresults
    y_fitted = model.fittedvalues
    X_fitted = model.model.exog
    X_fitted = sm.add_constant(X_fitted[:, 1:])

    # TEST 1: Shapiro-Wilk test to test normality of residual
    shapiro_test = shapiro(residuals)
    print ()
    print("--------------------")
    print(f"Shapiro-Wilk test for normality of residual: W={shapiro_test[0]}, p-value={shapiro_test[1]}")

    # If W is close to 1 and the p-value is greater than 0.05: The data likely comes from a normal distribution, and any deviation from normality is not statistically significant.
    # Interpretation
    if shapiro_test[1] < 0.05:
        print("Residual is not normally distributed - BAD.")
    else:
        print("Residual is normally distributed - GOOD.")


    # TEST 2: Breusch-Pagan test to test homoscedacity (constant variance of residual / error)
    bp_test = het_breuschpagan(residuals, X_fitted)
    bp_test_statistic = bp_test[0]
    bp_test_pvalue = bp_test[1]

    print ("--------------------")
    print(f'Breusch-Pagan test statistic: {bp_test_statistic}')
    print(f'Breusch-Pagan test p-value: {bp_test_pvalue}')

    # Interpretation
    if bp_test_pvalue < 0.05:
        print("Evidence of heteroscedasticity - BAD.")
    else:
        print("No evidence of heteroscedasticity - GOOD.")

def plot_histogram(df,plot_labels,log_transform_vars_flag):

    y_log = 'log_damage_gdp_weighted'
    x_log = ["log_vpd","log_pdforest","log_hvi"]
    y_non_log = 'damage_gdp_weighted'
    x_non_log = ["summer_vpd", "pop_forest", "hvi"]

    """
    PLOT
    """
    sns.set_theme(style="darkgrid")
    label_fontsize = 14
    tick_fontsize = 13

    fig = plt.figure(figsize=(10, 7))

    for n in range(0, 4):

        ax = fig.add_subplot(2, 2, n+1)

        if log_transform_vars_flag == False:

            if n == 0:
                to_plot = df[y_non_log]
                x_lab = plot_labels[y_non_log]
            else:
                xvar = x_non_log[n-1]
                x_lab = plot_labels[xvar]
                to_plot = df[xvar]

        if log_transform_vars_flag == True:

            if n == 0:
                to_plot = df[y_log]
                x_lab = plot_labels[y_log]
            else:
                xvar = x_log[n-1]
                x_lab = plot_labels[xvar]
                to_plot = df[xvar]

        ax = sns.histplot(data=to_plot, bins=15, edgecolor='white', kde=True, zorder=3)

        plt.xlabel(x_lab, fontsize=label_fontsize)  # Set x-label and fontsize
        plt.ylabel('Frequency', fontsize=label_fontsize)  # Set y-label and fontsize
        plt.xticks(fontsize=tick_fontsize)  # Set x-ticks fontsize
        plt.yticks(fontsize=tick_fontsize)  # Set y-ticks fontsize

    # Displaying the plot
    plt.tight_layout(pad=1.0)
    plt.show()

def plot_scatter(df,plot_labels,log_transform_vars_flag):

    y_log = 'log_damage_gdp_weighted'
    x_log = ["log_vpd","log_pdforest","log_hvi"]
    y_non_log = 'damage_gdp_weighted'
    x_non_log = ["summer_vpd", "pop_forest", "hvi"]

    """
    PLOT
    """
    title_fontsize = 16
    label_fontsize = 14
    legend_fontsize = 15
    tick_fontsize = 13

    fig = plt.figure(figsize=(14, 4))

    for n in range(0, 3):

        ax = fig.add_subplot(1, 3, n + 1)

        if log_transform_vars_flag == True:

            xvar = x_log[n]
            x = df[xvar]
            y = df[y_log]

            x_lab = plot_labels[xvar]
            y_lab = plot_labels[y_log]

        if log_transform_vars_flag == False:

            xvar = x_non_log[n]
            x = df[xvar]
            y = df[y_non_log]

            x_lab = plot_labels[xvar]
            y_lab = plot_labels[y_non_log]


        plt.scatter(x, y, s=50, alpha=0.8)

        plt.xlabel(x_lab, fontsize=label_fontsize)  # Set x-label and fontsize
        plt.ylabel(y_lab, fontsize=label_fontsize)  # Set y-label and fontsize
        plt.xticks(fontsize=tick_fontsize)  # Set x-ticks fontsize
        plt.yticks(fontsize=tick_fontsize)  # Set y-ticks fontsize

        # Plot the regression line
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        x_range = np.linspace(x.min(), x.max(), 100)
        y_range = slope * x_range + intercept
        plt.plot(x_range, y_range, color='black')

    # Displaying the plot
    plt.tight_layout(pad=2.0)
    plt.show()

def main():

    base_path = f'../../data'
    training_data_file = f'{base_path}/historical/TRAINING_DATA.csv'
    model_file = f'{base_path}/historical/model.pickle'

    df = pd.read_csv(training_data_file)
    df = df.dropna(subset=['log_damage_gdp_weighted'])

    plot_labels = {
        'damage_gdp_weighted':r'$cost$'
        , 'log_damage_gdp_weighted': r'$log(cost)$'
        , 'log_damage_gdp_weighted_std': r'$log(cost)$*'
        , 'log_vpd': r'$log(VPD_{fs})$'
        , 'log_pdforest': r'$log(PD_{forest})$'
        , 'log_hvi': r'$log(HVI)$'
        , 'log_vpd_std': r'$log(VPD_{fs})$*'
        , 'log_pdforest_std': r'$log(PD_{forest})$*'
        , 'log_hvi_std': r'$log(HVI)$*'
        , 'summer_vpd': r'VPD$_{fs}$'
        , 'pop_forest': r'PD$_{forest}$'
        , 'hvi': r'HVI'
               }

    """
    CHECK RESIDUAL OF FITTED MODEL
    """
    check_model_residual(model_file)

    """
    PLOT HISTOGRAM
    """
    log_transform_vars_flag = True # set to True if plot histogram for log-transformed X and y
    plot_histogram(df,plot_labels,log_transform_vars_flag)

    """
    PLOT X VS Y SCATTER PLOTS
    """
    log_transform_vars_flag = True  # set to True if plot histogram for log-transformed X and y
    plot_scatter(df,plot_labels,log_transform_vars_flag)

if __name__ == '__main__':

    main()