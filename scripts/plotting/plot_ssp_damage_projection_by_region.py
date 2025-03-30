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

base_path = f"../../data"
input_file = f"{base_path}/ssp/OUTPUT_SSP_PREDICTION.csv"

y_var = "damage_gdp_weighted"
show_global_mean = True
limit_forest_pct = True
forest_threshold_pct = 0.05

ssps = ['ssp1','ssp2','ssp3']
years = [2030, 2050, 2070]
regions = ["AFR","APC","DEV","EEA","LAM","MEA"]

"""
GET DATA
"""
df = pd.read_csv(input_file)
# Drop rows where esm is "MM"
df = df[df['esm'] != 'MM']
df = df[df['year'].isin(years)]
df = df[df['region_ar6_6'].isin(regions)]
df = df[df['ssp'].isin(ssps)]
df = df.dropna(subset=[y_var])

if limit_forest_pct == True:
    df = df[df['forest_pct'] >= forest_threshold_pct]

num_cntry = df['iso'].nunique()
print("NUM SSP COUNTRIES:", num_cntry)

# Function to calculate mean and 95% CI
def mean_ci(group_data):
    # First average across countries for each ESM
    esm_means = group_data.groupby('esm')[y_var].mean()
    
    # Calculate mean across ESMs
    mean = esm_means.mean()
    
    if len(esm_means) > 1:
        # Calculate standard deviation across ESMs
        std = esm_means.std(ddof=1)
        n = len(esm_means)  # number of ESMs
        sem = std / np.sqrt(n)
        
        # Calculate 95% CI width
        ci = stats.t.interval(confidence=0.95, df=n-1, loc=0, scale=sem)[1]
    else:
        ci = 0
    return pd.Series({'mean': mean, 'ci': ci})

# Calculate mean and CI for each region, ssp, and year
summary = df.groupby(['year', 'ssp', 'region_ar6_6']).apply(mean_ci).unstack()

# Extract means and CIs
means = summary['mean']
cis = summary['ci']

# Print global mean values for each year and SSP
print("\nGlobal Mean Values (%GDP):")
print("-" * 50)
print(f"{'Year':<10}{'SSP':<10}{'Mean':<10}")
print("-" * 50)
for year in years:
    for ssp in ssps:
        df_filtered = df[(df['ssp'] == ssp) & (df['year'] == year)]
        # First average by country for each ESM, then average across ESMs
        esm_means = df_filtered.groupby(['esm', 'iso'])[y_var].mean().groupby('esm').mean()
        overall_mean = esm_means.mean()
        print(f"{year:<10}{ssp:<10}{overall_mean:.4f}")
print("-" * 50)

"""
PLOT
"""

import seaborn as sns
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(3, 3, figsize=(10, 9), sharex=True, sharey=True)

title_fontsize = 14
label_fontsize = 13
tick_fontsize = 11
legend_fontsize = 13

# cmap = "Accent"
# colors = sns.color_palette(cmap, len(regions))

lab_colors = {
    'AFR': '#7FC97F', # green, (0.4980392156862745, 0.788235294117647, 0.4980392156862745)
    'APC': '#BEAED4', # purple,  (0.7450980392156863, 0.6823529411764706, 0.8313725490196079)
    'DEV': '#FDC086', # coral/orange, (0.9921568627450981, 0.7529411764705882, 0.5254901960784314)
    #'EEA': '#FFFF99', # yellow, (1.0, 1.0, 0.6),
    'EEA': 'mediumturquoise', # yellow, (1.0, 1.0, 0.6),
    'LAM': '#386CB0', # blue, (0.2196078431372549, 0.4235294117647059, 0.6901960784313725)
    'MEA': '#F0027F', # pink, (0.9411764705882353, 0.00784313725490196, 0.4980392156862745)
}

ssp_labels = utils.get_ssp_labels(ssps)
labels_alphabets = utils.generate_alphabet_list(9,option="lower")

n = -1
for i, year in enumerate(years):
    for j, ssp in enumerate(ssps):
        ax = axes[i, j]

        # Get the means and CIs for the current subplot
        mean_values = means.loc[year, ssp]
        ci_values = cis.loc[year, ssp]

        # Calculate global mean across all regions for this ssp and year
        if show_global_mean:
            df_filtered = df[(df['ssp'] == ssp) & (df['year'] == year)]
            # Calculate mean of means for each country to avoid giving more weight to countries with more ESMs
            country_means = df_filtered.groupby('iso')[y_var].mean()
            overall_mean = country_means.mean()

        for k, region in enumerate(regions):
            ax.barh(region, mean_values[region], xerr=ci_values[region], capsize=5, color=lab_colors[region],
                    zorder=3,
                    error_kw=dict(ecolor='grey', capthick=1, capsize=3))

        # Add vertical line for the overall mean value
        if show_global_mean == True:
            ax.axvline(overall_mean, color='darkblue', linewidth=1.5, zorder=4)

        # Set titles and labels
        if i == 0:
            ax.set_title(ssp_labels[j], fontsize=title_fontsize, weight="bold")
        if j == 0:
            ax.set_ylabel(f"{year}", fontsize=title_fontsize, fontweight='bold', labelpad=15)

        # Only set xlabel for the last row
        if i == 2:
            ax.set_xlabel('%GDP', fontsize=label_fontsize)
        else:
            ax.set_xlabel('')  # Clear xlabel for the first and second rows

        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labelsize=tick_fontsize)
        ax.grid(True, zorder=0, color="whitesmoke")

        n += 1

        plt.text(0.88, .95, f"({labels_alphabets[n]})", ha='left', va='top', transform=ax.transAxes, fontsize=label_fontsize)

fig.subplots_adjust(left=0.1,
                    bottom=0.07,
                    right=0.98,
                    top=0.95,
                    wspace=0.1,
                    hspace=0.2)

plt.show()