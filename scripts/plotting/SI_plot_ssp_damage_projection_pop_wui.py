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
base_path = "../../data"
input_file_pop_forest = f"{base_path}/ssp/OUTPUT_SSP_PREDICTION.csv"
input_files_pop_wui = f"{base_path}/ssp/pop_wui/OUTPUT_SSP_PREDICTION.csv"

y_var = "damage_gdp_weighted"
limit_forest_pct = True
forest_threshold_pct = 0.05

ssps = ['ssp1', 'ssp2', 'ssp3']
years = [2030, 2040, 2050, 2060, 2070]


ssp_labels = utils.get_ssp_labels(ssps)
labels_alphabets = utils.generate_alphabet_list(6, option="lower")

"""
LOAD DATA
"""
df_pop_forest = pd.read_csv(input_file_pop_forest)
df_pop_wui = pd.read_csv(input_files_pop_wui)

# Drop rows where esm is "MM"
df_pop_forest = df_pop_forest[df_pop_forest['esm'] != 'MM']
df_pop_wui = df_pop_wui[df_pop_wui['esm'] != 'MM']

# Apply filters to all dataframes
def filter_df(df):
    if limit_forest_pct:
        df = df[df['forest_pct'] >= forest_threshold_pct]
    df = df[df['year'].isin(years)]
    return df

df_pop_forest = filter_df(df_pop_forest)
df_pop_wui = filter_df(df_pop_wui)

def calculate_95_ci(data):
    """Calculate 95% confidence interval across ESMs after averaging across countries"""
    # First average across countries for each ESM
    esm_means = data.groupby('esm')[y_var].mean()
    
    if len(esm_means) > 1:
        # Calculate standard error across ESMs
        std = esm_means.std(ddof=1)
        n = len(esm_means)  # number of ESMs
        sem = std / np.sqrt(n)
        
        # Calculate 95% CI using t-distribution
        t_value = stats.t.ppf(0.975, df=n-1)  # Two-tailed 95% CI
        ci = t_value * sem
        mean = esm_means.mean()
    else:
        return np.nan, np.nan
    
    return mean, ci

"""
CALCULATE STATISTICS
"""
# Function to calculate statistics for each year-ssp combination
def calculate_stats(df, ssp):
    stats_list = []
    for year in years:
        # Get all data points for this year/ssp combination
        year_data = df[(df['year'] == year) & (df['ssp'] == ssp)]
        
        # Calculate mean and CI using ESM-level averaging
        mean, ci = calculate_95_ci(year_data)
        
        if not np.isnan(mean):  # Only add non-NaN values
            stats_list.append({
                'year': year,
                'mean': mean,
                'ci': ci
            })
    return pd.DataFrame(stats_list)

"""
PLOT
"""
import seaborn as sns
sns.set_theme(style="whitegrid")

# Set up colors
colors = {'pop_forest': '#8FBC8F',  # Light sage green
          'pop_wui': 'goldenrod'}     # Light coral pink

fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)

title_fontsize = 14
label_fontsize = 13
tick_fontsize = 13
legend_fontsize = 13

# Plot parameters
ylim_bottom = 0.2  # Set your desired bottom limit
ylim_top = 2.2    # Set your desired top limit
x_offset = 0.3  # Offset for markers to prevent overlap

# Plot for each SSP
for i, (ax, ssp) in enumerate(zip(axes, ssps)):
    # Calculate statistics for both datasets
    stats_forest = calculate_stats(df_pop_forest, ssp)
    stats_wui = calculate_stats(df_pop_wui, ssp)
    
    # Plot lines and points with offset
    ax.plot(stats_forest['year'] - x_offset, stats_forest['mean'], color=colors['pop_forest'], 
            label= r'${PD_{forest}}$', marker='o',markersize=4)
    ax.plot(stats_wui['year'] + x_offset, stats_wui['mean'], color=colors['pop_wui'],
            label= r'${pop_{wui}}$', marker='o',markersize=4)
    
    # Add error bars with thicker lines for consistency
    ax.errorbar(stats_forest['year'] - x_offset, stats_forest['mean'], yerr=stats_forest['ci'],
                color=colors['pop_forest'], capsize=2, capthick=0.8, elinewidth=0.8, ls='none',alpha=0.8)
    ax.errorbar(stats_wui['year'] + x_offset, stats_wui['mean'], yerr=stats_wui['ci'],
                color=colors['pop_wui'], capsize=2, capthick=0.8, elinewidth=0.8, ls='none',alpha=0.8)
    
    # Customize plot
    ax.set_title(f'({labels_alphabets[i]}) {ssp_labels[i]}', fontsize=title_fontsize, weight="bold")
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.tick_params(axis='y', which='both', labelleft=True)  # Show y-axis labels for all subplots
    ax.grid(True, alpha=0.3)
    
    # Set ylim
    ax.set_ylim(ylim_bottom, ylim_top)
    
    # Add labels
    if i == 0:  # Only add y-label to first subplot
        ax.set_ylabel(f'Mean damage [%GDP]', fontsize=label_fontsize)
    ax.set_xlabel('Year', fontsize=label_fontsize)
    
    if i == 2:
        # Add legend outside the plots
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), 
                     fontsize=legend_fontsize)

# Adjust layout
plt.subplots_adjust(left=0.065,
                    bottom=0.2,
                    right=0.87,
                    top=0.88,
                    wspace=0.18,
                    hspace=0.33)

plt.show()
