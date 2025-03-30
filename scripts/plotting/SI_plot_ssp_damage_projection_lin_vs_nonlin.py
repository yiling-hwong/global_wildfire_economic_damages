# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.append('../')
import utils

import warnings
warnings.filterwarnings(action='ignore')

"""
PARAMETERS
"""
nonlin_terms = ["vpd2","pd2","hvi2","vpd2_pd2","vpd2_hvi2","pd2_hvi2"]

base_path = "../../data"
input_file_lin = f"{base_path}/ssp/OUTPUT_SSP_PREDICTION.csv"
input_files_nonlin = base_path + "/ssp/nonlin/OUTPUT_SSP_PREDICTION_{}.csv"

y_var = "damage_gdp_weighted"
limit_forest_pct = True
forest_threshold_pct = 0.05

ssps = ['ssp1','ssp2','ssp3']
years = [2030,2040,2050,2060,2070]

ssp_labels = utils.get_ssp_labels(ssps)
labels_alphabets = utils.generate_alphabet_list(6,option="lower")

"""
LOAD DATA
"""
# Load linear model predictions
df_lin = pd.read_csv(input_file_lin)
df_lin = df_lin[df_lin['esm'] != 'MM']

# Load non-linear model predictions for each term
nonlin_dfs = {}
for term in nonlin_terms:
    file_path = input_files_nonlin.format(term)
    nonlin_dfs[term] = pd.read_csv(file_path)
    nonlin_dfs[term] = nonlin_dfs[term][nonlin_dfs[term]['esm'] != 'MM']

# Apply filters to all dataframes
def filter_df(df):
    if limit_forest_pct:
        df = df[df['forest_pct'] >= forest_threshold_pct]
    df = df[df['year'].isin(years)]
    return df

df_lin = filter_df(df_lin)
for term in nonlin_terms:
    nonlin_dfs[term] = filter_df(nonlin_dfs[term])

def calculate_95_ci(data):
    """Calculate 95% confidence interval using t-distribution"""
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    if len(data) < 2:
        return np.nan, np.nan
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard Error of Mean
    ci = stats.t.interval(alpha=0.95, df=n-1, loc=mean, scale=sem)
    return mean, float((ci[1] - ci[0])/2)  # Return mean and half the CI width

"""
PLOT
"""
import seaborn as sns
sns.set_theme(style="whitegrid")

# First collect all y values to find global min and max
all_y_values = []
all_yerr_values = []

for col in range(len(ssps)):
    # Get linear data
    linear_y = []
    linear_yerr = []
    for year in years:
        year_data = df_lin[(df_lin['ssp'] == ssps[col]) & 
                          (df_lin['year'] == year)]
        mean, ci = calculate_95_ci(year_data[y_var])
        linear_y.append(mean)
        linear_yerr.append(ci)
    
    all_y_values.extend(linear_y)
    all_yerr_values.extend(linear_yerr)
    
    # Get nonlinear data
    for term in nonlin_terms:
        term_y = []
        term_yerr = []
        for year in years:
            year_data = nonlin_dfs[term][(nonlin_dfs[term]['ssp'] == ssps[col]) & 
                                       (nonlin_dfs[term]['year'] == year)]
            mean, ci = calculate_95_ci(year_data[y_var])
            term_y.append(mean)
            term_yerr.append(ci)
        all_y_values.extend(term_y)
        all_yerr_values.extend(term_yerr)

# Calculate global y limits with some padding
y_min = min(np.array(all_y_values) - np.array(all_yerr_values))
y_max = max(np.array(all_y_values) + np.array(all_yerr_values))
y_range = y_max - y_min
#global_ymin = y_min - 0.05 * y_range
#global_ymax = y_max + 0.05 * y_range
global_ymin = 0.2
global_ymax = 2.2

# Set up the figure
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=True)

# Font sizes
title_fontsize = 14
label_fontsize = 13
tick_fontsize = 13
legend_fontsize = 13

# Color palette for different models
palette = sns.color_palette("husl", n_colors=len(nonlin_terms) + 1)
model_colors = {'Linear': 'dimgrey'}
for i, term in enumerate(nonlin_terms, 1):
    model_colors[f'{term}'] = palette[i]

# Split nonlin_terms into two groups
first_row_terms = nonlin_terms[:3]
second_row_terms = nonlin_terms[3:]

# Create plots
for col, ssp in enumerate(ssps):
    for row in range(2):
        current_terms = first_row_terms if row == 0 else second_row_terms
        ax = axes[row, col]
        
        # Plot linear model with error bars
        linear_y = []
        linear_yerr = []
        for year in years:
            year_data = df_lin[(df_lin['ssp'] == ssp) & 
                              (df_lin['year'] == year)]
            mean, ci = calculate_95_ci(year_data[y_var])
            linear_y.append(mean)
            linear_yerr.append(ci)
        
        # Set up offsets for markers
        offset = 3.5  # Slightly increased base offset
        n_terms = len(current_terms) + 1  # Include linear term
        # Create evenly spaced offsets centered around 0
        x_offsets = np.linspace(-offset/2, offset/2, n_terms)
        
        # Plot linear model error bars and line
        ax.errorbar(np.array(years) + x_offsets[0], linear_y, 
                   yerr=linear_yerr,
                   color=model_colors['Linear'], 
                   linestyle='none',  
                   marker='o', markersize=4,
                   capsize=2, capthick=0.5,   # Thinner error bars
                   elinewidth=0.5,            # Thinner error bars
                   label='_nolegend_',alpha=0.8)
        
        # Add connecting line for linear model
        ax.plot(np.array(years) + x_offsets[0], linear_y, 
               color=model_colors['Linear'],
               linestyle='dotted',
               marker='o', markersize=4,
               label='Linear' if col == 2 else '_nolegend_')

        # Plot non-linear terms for this row
        for i, term in enumerate(current_terms, 1):
            term_y = []
            term_yerr = []
            for year in years:
                year_data = nonlin_dfs[term][(nonlin_dfs[term]['ssp'] == ssp) & 
                                           (nonlin_dfs[term]['year'] == year)]
                mean, ci = calculate_95_ci(year_data[y_var])
                term_y.append(mean)
                term_yerr.append(ci)
            
            # Plot error bars and line for non-linear term
            ax.errorbar(np.array(years) + x_offsets[i], term_y, 
                       yerr=term_yerr,
                       color=model_colors[term], 
                       linestyle='none',
                       marker='o', markersize=4,
                       capsize=2, capthick=0.5,   # Thinner error bars
                       elinewidth=0.5,            # Thinner error bars
                       label='_nolegend_',alpha=0.8)
            
            # Add connecting line for non-linear term
            ax.plot(np.array(years) + x_offsets[i], term_y, 
                   color=model_colors[term],
                   linestyle='-',
                   marker='o', markersize=4,
                   label=term if col == 2 else '_nolegend_')

        # Set x-ticks to show only the actual years
        ax.set_xticks(years)
        ax.set_xticklabels(years)

        # Set y-limits for all subplots
        ax.set_ylim(global_ymin, global_ymax)

        # Set title and labels
        plot_idx = row * 3 + col
        ssp_label = ssp_labels[col]  # Get SSP label from the imported list
        ax.set_title(f'({labels_alphabets[plot_idx]}) {ssp_label}', 
                    fontsize=title_fontsize, weight="bold")
            
        # Only show x-label in second row
        if row == 1:
            ax.set_xlabel('Year', fontsize=label_fontsize)
        else:
            ax.set_xlabel('')
            
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.grid(True, color="gainsboro")
        
        if col == 0:
            ax.set_ylabel(f'Mean damage [%GDP]', fontsize=label_fontsize)
        
        # Handle legend
        if col == 2:  # Last column
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), 
                     fontsize=legend_fontsize)
        else:
            ax.legend().remove()

# Adjust layout
fig.subplots_adjust(left=0.06,
                   bottom=0.1,
                   right=0.87,
                   top=0.93,
                   wspace=0.07,
                   hspace=0.33)

plt.show()
