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
y_var = "damage_gdp_weighted"
vars_to_plot = [y_var,"summer_vpd","pop_forest","hvi"]

base_path = f"../../data"
input_file_hist = f"{base_path}/historical/TRAINING_DATA.csv"
country_shape_file = f"{base_path}/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

"""
LOAD DATA
"""
df = pd.read_csv(input_file_hist)
df = df.dropna(subset=[y_var])

y_labels = utils.get_plot_titles_and_labels(plot_option="label",average_flag=False)
titles = utils.get_plot_titles_and_labels(plot_option="title",average_flag=False)

# Load the world map shapefile
world = gpd.read_file(country_shape_file)
world = world.merge(df, how="left", left_on="ADM0_A3", right_on="iso")

"""
PLOT
"""

title_fontsize = 14
label_fontsize = 13
tick_fontsize = 12
legend_fontsize = 12
cmap = "YlOrRd"

from matplotlib.colors import LinearSegmentedColormap
colors = plt.cm.YlOrRd(np.linspace(0, 1, 35)) #viridis,plasma,YlOrRd
#colors = plt.cm.plasma(np.linspace(0, 1, 35)) #viridis,plasma,YlOrRd
new_colors = colors[5:]  # Exclude the last color (white)
cmap = LinearSegmentedColormap.from_list('custom_hot', new_colors)

fig = plt.figure(figsize=(10, 6))

num_row = 2
num_col = 2

labels_alphabets = utils.generate_alphabet_list(len(vars_to_plot),option="lower")

for index,var in enumerate(vars_to_plot):

    ax1 = fig.add_subplot(num_row, num_col, index + 1)

    var = vars_to_plot[index]
    print ("------------------------")
    print(var)
    y_label = y_labels[var]
    title = f'({labels_alphabets[index]}) {titles[var]}'

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    # Reproject world geometries to Robinson (EPSG:54030)
    robinson = ccrs.Robinson().proj4_init
    world = world.to_crs(robinson)
    world.boundary.plot(ax=ax1, linewidth=0.2, edgecolor='gray')

    if var == "pop_forest":

        vmin = world[var].min() *1.1 #1.2, 1.5
        vmax = world[var].max() *0.1 #0.8, 0.5

    elif var == "damage_gdp_weighted":

        vmin = world[var].min() *1.1 #1.2, 1.5
        vmax = world[var].max() *0.5 #0.8, 0.5

    else:
        vmin = world[var].min() *1.2
        vmax = world[var].max() *0.8


    plot = world.plot(column=var,
                      ax=ax1,
                      legend=True,
                      cax=cax,
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax,
               legend_kwds={'label': f"{y_label}",
                            'orientation': "vertical",
                            'extend':'both'},
               missing_kwds={"color": "white", "label": "No Data"})


    # Set label and tick size of colorbar
    cax.yaxis.label.set_size(label_fontsize)
    cax.tick_params(labelsize=tick_fontsize)

    ax1.set_title(title, fontdict={'fontsize': title_fontsize}, pad=12, weight="bold")
    ax1.set_axis_off()

# top_10_pop_forest = world.nlargest(10, 'pop_forest')
# print ("-----------")
# print ("TOP 10 pop_forest:")
# print (top_10_pop_forest[['iso','pop_forest']])

# -----------------------------------------
fig.subplots_adjust(left=0.001,
                    bottom=0.06,
                    right=0.93,
                    top=0.92,
                    wspace=0.07,  # 0.1
                    hspace=0.5)  # 0.85

"""
wspace and hspace specify the space reserved between Matplotlib subplots. They are the fractions of axis width and height, respectively.
left, right, top and bottom parameters specify four sides of the subplots’ positions. They are the fractions of the width and height of the figure.
top and bottom should add up to 1.0
"""

# Show plot
plt.show()
