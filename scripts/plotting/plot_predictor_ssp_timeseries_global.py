# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import utils

import warnings
warnings.filterwarnings(action='ignore')

"""
PARAMETERS
"""
base_path = f"../../data"
input_file_ssp = f"{base_path}/ssp/OUTPUT_SSP_PREDICTION.csv"

limit_forest_pct = True
forest_threshold_pct = 0.05
vars = ["summer_vpd","pop_forest","hvi"]
ssps = ['ssp1','ssp2','ssp3']
years = [2030, 2040, 2050, 2060, 2070]
regions = ["AFR","APC","DEV","EEA","LAM","MEA"]

ssp_labels = utils.get_ssp_labels(ssps)
y_labels = utils.get_plot_titles_and_labels(plot_option="label",average_flag=True)
titles = utils.get_plot_titles_and_labels(plot_option="title",average_flag=True)

"""
LOAD AND EXTRACT DATA
"""
df_ssp = pd.read_csv(input_file_ssp)

# Print unique ESMs before filtering
print("\nUnique ESMs before filtering:")
print(df_ssp['esm'].unique())

# Get predictors in original scale (exponentiate logged column)
df_ssp["summer_vpd"] = np.exp(df_ssp["log_vpd"])
df_ssp["pop_forest"] = np.exp(df_ssp["log_pdforest"])
df_ssp["hvi"] = np.exp(df_ssp["log_hvi"])

# Filter out MM model and apply initial filters
df_ssp = df_ssp[df_ssp['esm'] != 'MM']
df_ssp = df_ssp[df_ssp['year'].isin(years)]
df_ssp = df_ssp[df_ssp['region_ar6_6'].isin(regions)]
df_ssp = df_ssp[df_ssp['ssp'].isin(ssps)]
df_ssp = df_ssp.dropna(subset=vars)

if limit_forest_pct == True:
    df_ssp = df_ssp[df_ssp['forest_pct'] >= forest_threshold_pct]

# Calculate means and confidence intervals
from scipy.stats import t
import numpy as np
import scipy.stats as stats

def calculate_95_ci(data):
    """Calculate 95% confidence interval using t-distribution"""
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    if len(data) < 2:
        return 0, 0  # Return 0 instead of nan
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    sem = std / np.sqrt(n)
    ci = t.interval(confidence=0.95, df=n-1, loc=mean, scale=sem)
    return mean, float((ci[1] - ci[0])/2)  # Return mean and half the CI width

# Calculate statistics for each variable
stats_dict = {}
for var in vars:
    # Calculate mean and CI for each year-ssp combination
    stats_list = []
    for (year, ssp), group in df_ssp.groupby(['year', 'ssp']):
        # First average across countries for each ESM
        esm_means = group.groupby('esm')[var].mean()
        
        # Calculate mean across ESMs
        mean = esm_means.mean()
        
        # Calculate standard deviation across ESMs
        std = esm_means.std(ddof=1)
        n = len(esm_means)  # number of ESMs
        sem = std / np.sqrt(n)
        
        # Calculate 95% CI width
        ci = t.interval(confidence=0.9, df=n-1, loc=0, scale=sem)[1]
            
        if not np.isnan(mean):  # Only add non-NaN values
            stats_list.append({
                'year': year,
                'ssp': ssp,
                'mean': mean,
                'ci': ci
            })
    
    stats_dict[var] = pd.DataFrame(stats_list)

cntry_ssp = df_ssp['iso'].unique()
print("NUM countries in SSP:", len(cntry_ssp))

# Print summary statistics for each variable
# print("\nSummary statistics for each variable:")
# for var in vars:
#     print(f"\n{var}:")
#     for year in years:
#         for ssp in ssps:
#             data = df_ssp[(df_ssp['year'] == year) & (df_ssp['ssp'] == ssp)][var]
#             print(f"Year {year}, {ssp}:")
#             print(f"  Mean: {data.mean():.4f}")
#             print(f"  Std:  {data.std():.4f}")
#             print(f"  Min:  {data.min():.4f}")
#             print(f"  Max:  {data.max():.4f}")
#             print(f"  N:    {len(data)}")

"""
PLOT
"""
import seaborn as sns
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharey=False)

title_fontsize = 14
label_fontsize = 13
tick_fontsize = 12
legend_fontsize = 12

colors = {'ssp1': 'olivedrab', 'ssp2': 'goldenrod','ssp3': 'lightcoral'}
labels_alphabets = utils.generate_alphabet_list(4,option="lower")

ymin_list = [10,100,0.15]
ymax_list = [20,270,0.45]

# First collect all y values to find global min and max
all_y_values = []
all_yerr_values = []

# Loop through each numerical column and plot the data
for i, var in enumerate(vars):
    ax = axes[i]
    y_label = y_labels[var]
    title = f"({labels_alphabets[i]}) {titles[var]}"
    
    stats_df = stats_dict[var]
    
    # Calculate y-axis limits
    ymin = ymin_list[i]
    ymax = ymax_list[i]

    # Set up offsets for markers if showing error bars for all variables
    offset = 0.8  # Increased base offset from 0.3 to 0.8
    x_offsets = np.linspace(-offset, offset, len(ssps))

    if i != 0:
        x_offsets = [0, 0, 0]
    
    # Plot for each SSP
    for j, ssp in enumerate(ssps):
        ssp_data = stats_df[stats_df['ssp'] == ssp].sort_values('year')

        # Plot error bars for all variables
        ax.errorbar(ssp_data['year'] + x_offsets[j], ssp_data['mean'], 
                   yerr=ssp_data['ci'],
                   color=colors[ssp], 
                   linestyle='none',  
                   marker='o', markersize=4,
                   capsize=2, capthick=0.8, elinewidth=0.8, label='_nolegend_', alpha=0.8)
        

        # Add connecting line
        ax.plot(ssp_data['year'] + x_offsets[j], ssp_data['mean'], 
               color=colors[ssp],
               linestyle='solid',
               marker='o', markersize=4,
               label=ssp_labels[j] if i == 2 else '_nolegend_')
    
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=title_fontsize, weight="bold")
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.set_xlabel('Year', fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.grid(True, color="gainsboro")
    
    if i == 2:
        # Add a common legend outside of the figure on the right side
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ssp_labels, loc='center left', 
                 bbox_to_anchor=(1.0, 0.5), fontsize=label_fontsize)
    else:
        ax.legend().remove()

# -----------------------------------------
fig.subplots_adjust(left=0.055,
                   bottom=0.2,
                   right=0.885,
                   top=0.88,
                   wspace=0.3)

plt.show()
