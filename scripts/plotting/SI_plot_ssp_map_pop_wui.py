"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs

sys.path.append('../')
import utils

import warnings
warnings.filterwarnings(action='ignore')

"""
PARAMETERS
"""

vars = ["pop_forest"]
limit_forest_pct = True
forest_threshold_pct = 0.05

# Historical periods
pdforest_hist_period = {
    'start': 2010,
    'end': 2020
}

# Future periods
pdforest_future_period = {
    'start': 2060,
    'end': 2070
}

# Input files
base_path = f"../../data"
pdforest_file = f"{base_path}/ssp/raw/pop_wui.csv"
ssp_projection_file = f"{base_path}/ssp/OUTPUT_SSP_PREDICTION.csv"
world_shapefile = f"{base_path}/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

# SSPs to plot
ssps = ['ssp1', 'ssp2', 'ssp3']
ssp_labels = utils.get_ssp_labels(ssps)
y_labels = utils.get_plot_titles_and_labels(plot_option="label",average_flag=False)
titles = utils.get_plot_titles_and_labels(plot_option="title",average_flag=False)

"""
LOAD AND PROCESS DATA
"""
# Load world shapefile
world = gpd.read_file(world_shapefile)

# Filter countries
df_ssp = pd.read_csv(ssp_projection_file)
df_ssp = df_ssp.dropna(subset=['damage_gdp_weighted'])
if limit_forest_pct == True:
    df_ssp = df_ssp[df_ssp['forest_pct'] >= forest_threshold_pct]

ssp_countries = df_ssp["iso"].unique()
print ("NUM SSP countries:",len(ssp_countries))

#-------------------POP_WUI
df_pdforest = pd.read_csv(pdforest_file)
df_pdforest = df_pdforest[df_pdforest['ssp'].isin(ssps)]
df_pdforest = df_pdforest[df_pdforest['esm'] != 'MM']

if limit_forest_pct == True:
    df_pdforest = df_pdforest[df_pdforest["iso"].isin(ssp_countries)]

df_pdforest_hist_means = []
for ssp in ssps:
    df_hist_mean = df_pdforest[
        (df_pdforest['year'] >= pdforest_hist_period['start']) & 
        (df_pdforest['year'] <= pdforest_hist_period['end']) &
        (df_pdforest['ssp'] == ssp)
    ].groupby('iso')['pop_forest'].mean().reset_index()
    df_hist_mean['ssp'] = ssp
    df_pdforest_hist_means.append(df_hist_mean)
df_pdforest_hist_mean = pd.concat(df_pdforest_hist_means, ignore_index=True)

df_pdforest_ssp_mean = df_pdforest[
    (df_pdforest['year'] >= pdforest_future_period['start']) &
    (df_pdforest['year'] <= pdforest_future_period['end']) &
    (df_pdforest['ssp'].isin(ssps))
].groupby(['iso', 'ssp'])['pop_forest'].mean().reset_index()

df_pdforest_diff = pd.merge(
    df_pdforest_ssp_mean, 
    df_pdforest_hist_mean, 
    on=['iso', 'ssp'], 
    suffixes=('_future', '_hist')
)
df_pdforest_diff['diff'] = ((df_pdforest_diff['pop_forest_future'] - df_pdforest_diff['pop_forest_hist']) / df_pdforest_diff['pop_forest_hist']) * 100

"""
CREATE PLOT
"""
fig = plt.figure(figsize=(9, 2.5))

title_fontsize = 12
label_fontsize = 11
tick_fontsize = 11
legend_fontsize = 10

# Create big axes for SSP titles
for col in range(3):
    big_ax = fig.add_subplot(1, 3, col + 1)
    if col == 0:
        big_ax.set_title(f"SSP126", fontsize=title_fontsize+1, y=1.00, weight="bold")
    if col == 1:
        big_ax.set_title(f"SSP245", fontsize=title_fontsize+1, y=1.00, weight="bold")
    if col == 2:
        big_ax.set_title(f"SSP370", fontsize=title_fontsize+1, y=1.00, weight="bold")
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax._frameon = False

# Define color schemes
color_schemes = {
    'pdforest': {'vmin': -60, 'vmax': 60, 'cmap': plt.cm.BrBG_r}
}

# Map internal variable names to utils variable names
var_mapping = {
    'pdforest': 'pop_forest'
}

# Create row titles using the titles dictionary
row_titles = [titles[var_mapping['pdforest']]]
unit_labels = {'pop_forest': "%"}

# Create subplots
print ("Plotting...")
axes = []
row_axes = []
for col in range(3):
    ax = fig.add_subplot(1, 3, col + 1)
    row_axes.append(ax)
axes.append(row_axes)

scheme = color_schemes['pdforest']
norm = mpl.colors.TwoSlopeNorm(vmin=scheme['vmin'], vcenter=0, vmax=scheme['vmax'])

# Get the corresponding label from y_labels
var_label = y_labels[var_mapping['pdforest']]
unit_label = unit_labels[var_mapping['pdforest']]
main_label = var_label.split('[')[0].strip()

# Add row title at the top of the first subplot in the row
#fig.text(0.5, 0.72, row_titles[0], ha='center', va='center', fontsize=title_fontsize)

for col, ssp in enumerate(ssps):
    # Get data for this SSP
    ssp_data = df_pdforest_diff[df_pdforest_diff['ssp'] == ssp]
    
    # Get the axis
    ax = axes[0][col]

    # Merge with world shapefile
    world = gpd.read_file(world_shapefile)
    world = world.merge(ssp_data, how='left', left_on='ISO_A3', right_on='iso')

    # Reproject world geometries to Robinson (EPSG:54030)
    robinson = ccrs.Robinson().proj4_init
    world = world.to_crs(robinson)
    
    # Plot
    world.plot(column='diff',
                   ax=ax,
                   cmap=scheme['cmap'],
                   norm=norm,
                   missing_kwds={'color': 'white'})
    
    world.boundary.plot(ax=ax, linewidth=0.2, edgecolor='gray')
    ax.axis('off')
    
    # Add y-axis label for the first column
    if col == 0:
        ax.text(-0.03, 0.5, r'${pop_{wui}}$', transform=ax.transAxes,
               va='center', ha='center', rotation=90, fontsize=label_fontsize)

# Add colorbar
cax = fig.add_axes([0.91, 0.25, 0.008, 0.4]) #left,bottom,width,height
sm = plt.cm.ScalarMappable(cmap=scheme['cmap'], norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax, orientation='vertical', extend='both')
cbar.set_label(f'{unit_label}', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)

# Adjust layout
fig.subplots_adjust(left=0.04,  # Space for y-axis labels
                   bottom=0.00,
                   right=0.9,
                   top=0.78,  # Space for SSP titles
                   wspace=0.00,
                   hspace=0.0)

#plt.tight_layout()
plt.show()