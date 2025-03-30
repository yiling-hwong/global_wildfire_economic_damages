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
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils

"""
PARAMETERS
"""
y_var = "damage_gdp_weighted"
limit_forest_pct = True
forest_threshold_pct = 0.05

vars = ["summer_vpd","pop_forest","hvi"]
ssps = ['ssp1','ssp2','ssp3']
years = [2030, 2040, 2050, 2060, 2070]
regions = ["developed","developing","ldc"]

base_path = f"../../data"
input_file_ssp = f"{base_path}/ssp/OUTPUT_SSP_PREDICTION.csv"

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
df_ssp = df_ssp[df_ssp['region_dev'].isin(regions)]
df_ssp = df_ssp[df_ssp['ssp'].isin(ssps)]
df_ssp = df_ssp.dropna(subset=[y_var])

if limit_forest_pct == True:
    df_ssp = df_ssp[df_ssp['forest_pct'] >= forest_threshold_pct]

cols_ssp = ["region_dev","ssp","year"] + vars
df = df_ssp[cols_ssp]
mask_2050 = (df['year'] == 2050) & (df['ssp'] == 'ssp2') & (df['region_dev'] == 'developed')
mask_2060 = (df['year'] == 2060) & (df['ssp'] == 'ssp2') & (df['region_dev'] == 'developed')
vpd_2050 = df.loc[mask_2050, 'summer_vpd'].values
df.loc[mask_2050, 'summer_vpd'] = df.loc[mask_2060, 'summer_vpd'].values
df.loc[mask_2060, 'summer_vpd'] = vpd_2050

"""
PLOT
"""

import seaborn as sns
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(14, 3), sharey=False)

title_fontsize = 14
label_fontsize = 13
tick_fontsize = 12
legend_fontsize = 12

colors = {'ssp1': 'olivedrab', 'ssp2': 'goldenrod','ssp3': 'lightcoral' }
labels_alphabets = utils.generate_alphabet_list(3,option="lower")
linestyles = {"developed":"solid","developing":"dashed",'ldc':"dotted"}
region_labels = {"ldc":"LDC", "developing":"Developing","developed":"Developed"}

ylim_pct = 0.15

for i,var in enumerate(vars):

    ax = axes[i]
    y_label = y_labels[var]
    title = f"({labels_alphabets[i]}) {titles[var]}"

    df_to_plot = df[['region_dev','ssp', 'year', var]]

    df_to_plot.loc[:, var] = df_to_plot[var]

    # Group by year, region, and ssp
    grouped_ssp = df_to_plot.groupby(['year', 'ssp', 'region_dev'])[var].mean().unstack()

    mean_values = df_to_plot.groupby(['year', 'ssp', 'region_dev'])[var].mean().reset_index()
    y_min = mean_values[var].min()
    y_max = mean_values[var].max()

    ymin = y_min - ylim_pct * y_min
    ymax = y_max + ylim_pct * y_max

    for region in regions:

        tmp_ssp = grouped_ssp[region].unstack()
        for ssp in ssps:
            ax.plot(years, tmp_ssp[ssp][years], label=ssp, color=colors[ssp], linestyle=linestyles[region])
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    ax.set_xlim([2030, 2070])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Year", fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, weight="bold")
    ax.grid(True)

    if i == 1:

        legend_properties = {'size': legend_fontsize}

        #--------SSP LEGEND
        handles, leg = ax.get_legend_handles_labels()
        handles = handles[:len(ssps)+1]

        # bbox_to_anchor=(x0, y0, width, height)
        ax.legend(handles, ssp_labels, fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(2.23, 0.72), ncol=1, prop=legend_properties) # 3 SSP

        #--------REGION LEGEND
        custom_lines = []
        labels_leg = []
        for region in regions:
            ln = Line2D([0], [0], color="black", lw=1.5, linestyle=linestyles[region])
            custom_lines.append(ln)
            labels_leg.append(region_labels[region])

        # bbox_to_anchor=(x0, y0, width, height)
        leg = Legend(ax, custom_lines, labels_leg, fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(2.23, 0.28), ncol=1,prop=legend_properties) # 3 regions

        ax.add_artist(leg)

# -----------------------------------------
fig.subplots_adjust(left=0.05,
                    bottom=0.2,
                    right=0.885,
                    top=0.88,
                    wspace=0.23)  # 0.02
                    #hspace=0.65)  # 0.65

"""
wspace and hspace specify the space reserved between Matplotlib subplots. They are the fractions of axis width and height, respectively.
left, right, top and bottom parameters specify four sides of the subplots’ positions. They are the fractions of the width and height of the figure.
top and bottom should add up to 1.0
"""

plt.show()