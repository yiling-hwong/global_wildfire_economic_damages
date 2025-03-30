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

# Input files and parameters
base_path = "../../data"
pop_file = f"{base_path}/ssp/raw/pop_forest.csv"

# Years and SSPs to plot
years = [2030, 2070]

# Read data
df = pd.read_csv(pop_file)

# Filter for MM rows if esm column exists
if 'esm' in df.columns:
    df = df[df['esm'] == 'MM']

# Print unique values to debug
print("Unique regions:", df['region_ar6_6'].unique())

# Make sure SSP names match the data
ssps = df['ssp'].unique()

# Calculate global sum for each SSP-year combination
global_sums = df.groupby(['ssp', 'year'])['pop'].sum().reset_index()
global_sums['pop'] = global_sums['pop'] / 1e6  # Convert to millions

# Calculate regional sums
regional_sums = df.groupby(['ssp', 'year', 'region_ar6_6'])['pop'].sum().reset_index()
regional_sums['pop'] = regional_sums['pop'] / 1e6  # Convert to millions

"""
PLOT
"""

import seaborn as sns
sns.set_theme(style="whitegrid")

title_fontsize = 14
label_fontsize = 13
tick_fontsize = 13
legend_fontsize = 13

# Set up the plot
plt.figure(figsize=(10, 4))

# Define bar positions and widths
n_groups = len(df['region_ar6_6'].unique()) + 1  # Global + regions
n_bars = len(ssps) * len(years)    # SSPs × years
bar_width = 0.12
group_width = n_bars * bar_width
x = np.arange(n_groups)

# Define colors for each SSP
colors = {}
for i, ssp in enumerate(ssps):
    for j, year in enumerate(years):
        label = f"{ssp.upper()}-{year}"
        if i == 0:
            if j == 0:
                colors[label] = '#A6CEE3'  # Light blue
            else:
                colors[label] = '#1F78B4'  # Dark blue
        elif i == 1:
            if j == 0:
                colors[label] = '#B2DF8A'  # Light green
            else:
                colors[label] = '#33A02C'  # Dark green
        elif i == 2:
            if j == 0:
                colors[label] = '#FB9A99'  # Light red
            else:
                colors[label] = '#E31A1C'   # Dark red

# Plot bars for each SSP-year combination
bars = []
labels = []
for i, ssp in enumerate(ssps):
    for j, year in enumerate(years):

        # Get data for global
        global_data = global_sums[(global_sums['ssp'] == ssp) & 
                                 (global_sums['year'] == year)]['pop'].values[0]
        
        # Get data for regions
        region_data = regional_sums[(regional_sums['ssp'] == ssp) & 
                                   (regional_sums['year'] == year)].sort_values('region_ar6_6')
        
        # Combine global and regional data
        all_data = [global_data] + region_data['pop'].tolist()
        
        # Calculate bar positions
        bar_pos = x - group_width/2 + (i*len(years) + j + 0.5)*bar_width
        
        # Plot bars
        label = f"{ssp.upper()}-{year}"
        bar = plt.bar(bar_pos, all_data, bar_width, 
                     label=label, color=colors[label])
        bars.append(bar)
        labels.append(label)

# Customize the plot
#plt.xlabel('Region')
plt.ylabel('Population (millions)',fontsize=label_fontsize)

# Add custom grid
ax = plt.gca()
ax.grid(True, color="gainsboro")

# Set x-axis ticks
region_labels = ['Globe'] + sorted(df['region_ar6_6'].unique().tolist())
plt.xticks(x, region_labels, fontsize=tick_fontsize)

# Format y-axis to use regular numbers instead of scientific notation
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.yticks(fontsize=tick_fontsize)

# Add legend
plt.legend(bbox_to_anchor=(1.01, 0.8), loc='upper left',fontsize=legend_fontsize)

plt.tight_layout()
plt.show()