"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import cartopy.crs as ccrs

sys.path.append('../')
import utils

import warnings
warnings.filterwarnings(action='ignore')

"""
PARAMETERS
"""

vars = ["summer_vpd","pop_forest","hvi"]
limit_forest_pct = True
forest_threshold_pct = 0.05

# Historical periods
vpd_hist_period = {
    'start': 1990,
    'end': 2000
}

pdforest_hist_period = {
    'start': 2010,
    'end': 2020
}

hvi_hist_period = {
    'start': 1990,
    'end': 2000
}

# Future periods
vpd_future_period = {
    'start': 2050,
    'end': 2070
}

pdforest_future_period = {
    'start': 2050,
    'end': 2070
}

hvi_future_period = {
    'start': 2050,
    'end': 2070
}

# Input files
base_path = f"../../data"
vpd_hist_file = f"{base_path}/ssp/raw/vpd_precip_hist.csv"
vpd_ssp_file = f"{base_path}/ssp/raw/vpd_precip.csv"
pdforest_file = f"{base_path}/ssp/raw/pop_forest.csv"
hdi_hist_file = f"{base_path}/ssp/raw/hdi_hist.csv"
hdi_ssp_file = f"{base_path}/ssp/raw/hdi.csv"
ssp_projection_file = f"{base_path}/ssp/OUTPUT_SSP_PREDICTION.csv"
world_shapefile = f"{base_path}/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

# SSPs to plot
ssps = ['ssp1', 'ssp2', 'ssp3']
ssp_labels = utils.get_ssp_labels(ssps)
y_labels = utils.get_plot_titles_and_labels(plot_option="label",average_flag=False)
titles = utils.get_plot_titles_and_labels(plot_option="title",average_flag=False)
subplt_labels = utils.generate_alphabet_list(9, "lower")

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

#-------------------VPD
df_vpd_hist = pd.read_csv(vpd_hist_file)
df_vpd_hist = df_vpd_hist[df_vpd_hist['esm'] != 'MM']

if limit_forest_pct == True:
    df_vpd_hist = df_vpd_hist[df_vpd_hist["iso"].isin(ssp_countries)]

df_vpd_hist_mean = df_vpd_hist[
    (df_vpd_hist['year'] >= vpd_hist_period['start']) & 
    (df_vpd_hist['year'] <= vpd_hist_period['end'])
].groupby('iso')['summer_vpd'].mean().reset_index()

df_vpd_ssp = pd.read_csv(vpd_ssp_file)
df_vpd_ssp = df_vpd_ssp[df_vpd_ssp['ssp'].isin(ssps)]
df_vpd_ssp = df_vpd_ssp[df_vpd_ssp['esm'] != 'MM']

if limit_forest_pct == True:
    df_vpd_ssp = df_vpd_ssp[df_vpd_ssp["iso"].isin(ssp_countries)]

df_vpd_ssp_mean = df_vpd_ssp[
    (df_vpd_ssp['year'] >= vpd_future_period['start']) &
    (df_vpd_ssp['year'] <= vpd_future_period['end']) &
    (df_vpd_ssp['ssp'].isin(ssps))
].groupby(['iso', 'ssp'])['summer_vpd'].mean().reset_index()

df_vpd_diff = pd.merge(df_vpd_ssp_mean, df_vpd_hist_mean, on='iso', suffixes=('_ssp', '_hist'))
df_vpd_diff['diff'] = df_vpd_diff['summer_vpd_ssp'] - df_vpd_diff['summer_vpd_hist']

#-------------------POP_FOREST
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

#-------------------HVI
df_hvi_hist = pd.read_csv(hdi_hist_file)
df_hvi_hist['hdi'] = 1 - df_hvi_hist['hdi']  # Convert to HVI

if limit_forest_pct == True:
    df_hvi_hist = df_hvi_hist[df_hvi_hist["iso"].isin(ssp_countries)]

df_hvi_hist_mean = df_hvi_hist[
    (df_hvi_hist['year'] >= hvi_hist_period['start']) & 
    (df_hvi_hist['year'] <= hvi_hist_period['end'])
].groupby('iso')['hdi'].mean().reset_index()

df_hvi_ssp = pd.read_csv(hdi_ssp_file)
df_hvi_ssp['hdi'] = 1 - df_hvi_ssp['hdi']  # Convert to HVI
df_hvi_ssp = df_hvi_ssp[df_hvi_ssp['ssp'].isin(ssps)]

if limit_forest_pct == True:
    df_hvi_ssp = df_hvi_ssp[df_hvi_ssp["iso"].isin(ssp_countries)]

df_hvi_ssp_mean = df_hvi_ssp[
    #df_hvi_ssp['year'].isin(hvi_future_period['years'])
    (df_hvi_ssp['year'] >= hvi_future_period['start']) &
    (df_hvi_ssp['year'] <= hvi_future_period['end']) &
    (df_hvi_ssp['ssp'].isin(ssps))
].groupby(['iso', 'ssp'])['hdi'].mean().reset_index()

df_hvi_diff = pd.merge(df_hvi_ssp_mean, df_hvi_hist_mean, on='iso', suffixes=('_ssp', '_hist'))
df_hvi_diff['diff'] = df_hvi_diff['hdi_ssp'] - df_hvi_diff['hdi_hist']

"""
CREATE PLOT
"""
fig = plt.figure(figsize=(8, 6))

title_fontsize = 12
label_fontsize = 12
tick_fontsize = 11
legend_fontsize = 11

# Create big axes for SSP titles
for col in range(3):
    big_ax = fig.add_subplot(1, 3, col + 1)
    if col == 0:
        big_ax.set_title(f"SSP126", fontsize=title_fontsize+1, y=1.08, weight="bold")
    if col == 1:
        big_ax.set_title(f"SSP245", fontsize=title_fontsize+1, y=1.08, weight="bold")
    if col == 2:
        big_ax.set_title(f"SSP370", fontsize=title_fontsize+1, y=1.08, weight="bold")
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax._frameon = False

#proj = ccrs.Robinson()

# Define color schemes for each variable
color_schemes = {
    'vpd': {'vmin': -5, 'vmax': 5, 'cmap': plt.cm.BrBG_r},
    'pdforest': {'vmin': -60, 'vmax': 60, 'cmap': plt.cm.BrBG_r},
    'hvi': {'vmin': -0.2, 'vmax': 0.2, 'cmap': plt.cm.BrBG_r}
}

# Map our internal variable names to the utils variable names
var_mapping = {
    'vpd': 'summer_vpd',
    'pdforest': 'pop_forest',
    'hvi': 'hvi'
}

# Create row titles using the titles dictionary
row_titles = [titles[var_mapping[var]] for var in ['vpd', 'pdforest', 'hvi']]
unit_labels = {'summer_vpd':f"ΔhPa","pop_forest":"%","hvi":f"ΔHVI"}

# Create subplots
print ("Plotting...")
axes = []
for row in range(3):
    row_axes = []
    for col in range(3):
        #ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection=proj)
        ax = fig.add_subplot(3, 3, row * 3 + col + 1)
        row_axes.append(ax)
    axes.append(row_axes)

n = -1
for row, (var_name, df) in enumerate([('vpd', df_vpd_diff), ('pdforest', df_pdforest_diff), ('hvi', df_hvi_diff)]):
    scheme = color_schemes[var_name]
    norm = mpl.colors.TwoSlopeNorm(vmin=scheme['vmin'], vcenter=0, vmax=scheme['vmax'])
    
    # Get the corresponding label from y_labels
    var_label = y_labels[var_mapping[var_name]]
    unit_label = unit_labels[var_mapping[var_name]]
    main_label = var_label.split('[')[0].strip()

    # Add row title at the top of the first subplot in the row
    fig.text(0.5, 0.88 - row * 0.31, row_titles[row],
             ha='center', va='center', fontsize=title_fontsize)
    
    for col, ssp in enumerate(ssps):

        n += 1
        # Get data for this SSP
        ssp_data = df[df['ssp'] == ssp]
        
        # Get the axis
        ax = axes[row][col]

        # Merge with world shapefile
        world = gpd.read_file(world_shapefile)
        world = world.merge(ssp_data, how='left', left_on='ISO_A3', right_on='iso')

        # Reproject world geometries to Robinson (EPSG:54030)
        robinson = ccrs.Robinson().proj4_init
        world = world.to_crs(robinson)
        
        # Plot
        world.plot(column='diff',
                       ax=ax,
                       #transform=ccrs.PlateCarree(),
                       cmap=scheme['cmap'],
                       norm=norm,
                       missing_kwds={'color': 'white'})
        
        # Add coastlines
        world.boundary.plot(ax=ax, linewidth=0.2, edgecolor='gray')
        ax.axis('off')

        plt.text(0.0, .98, f"({subplt_labels[n]})", ha='left', va='top', transform=ax.transAxes,
                 fontsize=legend_fontsize)
        
        # Add y-axis label
        if col == 0:
            ax.text(-0.05, 0.5, main_label, transform=ax.transAxes,
                   va='center', ha='center', rotation=90, fontsize=label_fontsize)
        
        # Add colorbar with the unit part of the label (from square brackets)
        if col == 2:
            divider = make_axes_locatable(ax)
            cax = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)
            fig.add_axes(cax)
            sm = plt.cm.ScalarMappable(cmap=scheme['cmap'], norm=norm)
            cbar = plt.colorbar(sm, cax=cax, orientation='vertical',extend='both')
            cbar.set_label(label=f'{unit_label}',fontsize=label_fontsize-1)

# -----------------------------------------
fig.subplots_adjust(left=0.04,  # Space for y-axis labels
                    bottom=0.02,
                    right=0.9,
                    top=0.86,  # Increased space for SSP titles
                    wspace=0.00,
                    hspace=0.3)

plt.show()