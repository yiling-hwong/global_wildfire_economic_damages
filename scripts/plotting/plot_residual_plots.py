# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import os
import sys
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import linregress

import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('../')
import utils

import warnings
warnings.filterwarnings(action='ignore')

"""
PARAMETERS
"""
base_path = f"../../data"
training_data_file = f"{base_path}/historical/TRAINING_DATA.csv"
model_file = f"{base_path}/historical/model.pickle"

predictors = ["log_vpd_std", "log_pdforest_std", "log_hvi_std"]

df = pd.read_csv(training_data_file)

"""
GET FITTED MODEL AND RESIDUAL
"""

# Load the model from the pickle file
with open(model_file, 'rb') as file:
    model = pickle.load(file)

residuals = model.resid_response  # GLMresults

y_fitted = model.fittedvalues
X_fitted = model.model.exog
X_fitted = sm.add_constant(X_fitted[:, 1:])

"""
PLOT
"""

sns.set_theme(style="whitegrid")

fig = plt.figure(figsize=(8, 6))

title_fontsize = 14
label_fontsize = 13
tick_fontsize = 12
legend_fontsize = 12

scatter_size = 40
scatter_alpha = 0.6

num_row = 2
num_col = 2

x_labels = ["Predicted values", r'$log(VPD_{fs})$*', r'$log(PD_{forest})$*', r'$log(HVI)$*']
y_labels = ["Residuals",r'$Res. + \beta_{\mathrm{VPD_{fs}}}log(VPD_{fs})*$',
            r'$Res. + \beta_{\mathrm{PD_{forest}}}log(PD_{forest})*$',
            r'$Res. + \beta_{HVI}log(HVI)*$']
titles = ["Residual plot", "Partial residual plot", "Partial residual plot","Partial residual plot"]
labels_alphabets = utils.generate_alphabet_list(4,option="lower")

for n in range(4):

    ax = fig.add_subplot(num_row, num_col, n + 1)

    title = f"({labels_alphabets[n]}) {titles[n]}"

    """
    Residual plot
    """

    if n == 0:
        ax.scatter(y_fitted, residuals, s=scatter_size, alpha=scatter_alpha)  # GLMResults
        ax.axhline(0, color='dimgrey')

    else:
        plot = model.plot_partial_residuals(n, ax=ax)

        # PLOT REGRESSION LINE
        predictor = predictors[n - 1]
        x = df[predictor]
        y_pred = residuals + model.params[n] * x

        # Sort the values for a smoother line
        sorted_idx = np.argsort(x)
        x_sorted = x.iloc[sorted_idx]
        y_pred_sorted = y_pred[sorted_idx]

        x = x_sorted
        y = y_pred_sorted
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Plot the regression line
        x_range = np.linspace(x.min(), x.max(), 100)
        y_range = slope * x_range + intercept
        plt.plot(x_range, y_range, color='dimgrey')

    ax.set_xlabel(x_labels[n], fontsize=label_fontsize)
    ax.set_ylabel(y_labels[n], fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True, color="gainsboro")

# -----------------------------------------
fig.subplots_adjust(left=0.1,
                    bottom=0.13,
                    right=0.98,
                    top=0.92,
                    wspace=0.27, # 0.02
                    hspace=0.6)  # 0.65

"""
wspace and hspace specify the space reserved between Matplotlib subplots. They are the fractions of axis width and height, respectively.
left, right, top and bottom parameters specify four sides of the subplots’ positions. They are the fractions of the width and height of the figure.
top and bottom should add up to 1.0
"""

plt.show()
