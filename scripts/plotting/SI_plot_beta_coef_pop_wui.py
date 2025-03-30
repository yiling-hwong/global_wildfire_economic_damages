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
from matplotlib.ticker import LogLocator, FuncFormatter
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils

"""
PARAMETERS
"""
pop_forest_option = "pop_wui" #"pop_forest" for pop_forest = pop/forest_area; "pop_wui" for pop_forest = population in forested grid

if pop_forest_option == "pop_forest":
    pdforest_str = ""
if pop_forest_option == "pop_wui":
    pdforest_str = "pop_wui/"

base_path = f"../../data"
fitted_model = f'{base_path}/historical/{pdforest_str}model.pickle'
training_data_file = f'{base_path}/historical/{pdforest_str}TRAINING_DATA.csv'
y_hist_mean_std_file = f'{base_path}/historical/{pdforest_str}mean_SD_X_y.csv'

predictors = ["log_vpd", "log_pdforest", "log_hvi"]

# Load model
model_loaded = sm.load(fitted_model)

"""
GET BETA COEF FROM FITTED MODEL
"""
r_squared = model_loaded.pseudo_rsquared()
coefficients = model_loaded.params
se = model_loaded.bse
num_obs = model_loaded.nobs

coefficients = coefficients[1:]
se = se[1:]
conf_int = model_loaded.conf_int(alpha=0.05, cols=None)[1:]
p_vals = model_loaded.pvalues[1:]

coefficients_df = pd.DataFrame({'Coefficient': coefficients, 'Standard_Error': se})
confidence_intervals = conf_int
confidence_intervals_df = pd.DataFrame(confidence_intervals, columns=['CI_Lower', 'CI_Upper'])
coefficients_df = pd.concat([coefficients_df, confidence_intervals_df], axis=1)
coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)

"""
GET DATA FROM FITTED MODEL
"""
df = pd.read_csv(training_data_file)
df_y_hist_mean_std = pd.read_csv(y_hist_mean_std_file)

# Get X and y from training data
y_actual_usd = np.array(df["damage"])
y_actual_pct = np.array(df["damage_gdp_weighted"])
y_mean = df_y_hist_mean_std['mean_y_hist'].values
y_std = df_y_hist_mean_std['std_y_hist'].values

X_hist = df[predictors]
X_hist = sm.add_constant(X_hist)

scaler_std_X = StandardScaler()
scaler_std_y = StandardScaler()
X_hist = scaler_std_X.fit_transform(X_hist)
y_predicted = model_loaded.predict(X_hist)

# Transform to original scale
y_predicted = (y_predicted * y_std) + y_mean

# Exponentiate
y_predicted_pct = np.exp(y_predicted) # in %GDP
y_predicted_usd = np.array((y_predicted_pct / 100) * df["gdp"]) # in USD

y_compare = np.column_stack((y_actual_usd,y_predicted_usd))

df_to_plot = pd.DataFrame({"y_actual_pct":y_actual_pct,
                           "y_predicted_pct": y_predicted_pct,
                           "y_actual_usd":y_actual_usd,
                           "y_predicted_usd":y_predicted_usd})

df_to_plot["region_ar6_6"] = df["region_ar6_6"]

"""
PLOT
"""

sns.set_theme(style="whitegrid")

fig = plt.figure(figsize=(8, 7))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

title_fontsize = 16
label_fontsize = 14
legend_fontsize = 14
tick_fontsize = 14
markersize = 6
capsize = 5
cmap = "Accent" #Dark2, Accent

y_labels = {
    'log_vpd': r'$log(VPD_{fs})$*'
    , 'log_pdforest': r'$log(pop_{wui})$*'
    , 'log_hvi': r'$log(HVI)$*'
}

legend_labs = {
    'AFR': 'Africa',
    'APC': 'Asia & Developing Pacific',
    'DEV': 'Developed Countries',
    'EEA': 'East. Europe & West-Central Asia',
    'LAM': 'Latin America',
    'MEA': 'Middle East',
}

lab_colors = {
    'AFR': '#7FC97F', # green, (0.4980392156862745, 0.788235294117647, 0.4980392156862745)
    'APC': '#BEAED4', # purple,  (0.7450980392156863, 0.6823529411764706, 0.8313725490196079)
    'DEV': '#FDC086', # coral/orange, (0.9921568627450981, 0.7529411764705882, 0.5254901960784314)
    #'EEA': '#FFFF99', # yellow, (1.0, 1.0, 0.6),
    'EEA': 'mediumturquoise', # yellow, (1.0, 1.0, 0.6),
    'LAM': '#386CB0', # blue, (0.2196078431372549, 0.4235294117647059, 0.6901960784313725)
    'MEA': '#F0027F', # pink, (0.9411764705882353, 0.00784313725490196, 0.4980392156862745)
}


# RGB code: [[127.0, 201.0, 127.0], [190.0, 174.0, 212.0], [253.0, 192.0, 134.0], [255.0, 255.0, 153.0], [56.0, 108.0, 176.0], [240.0, 2.0, 127.0]]


#------------------------- BETA COEF
ax = fig.add_subplot(gs[0, :])

ax = plt.errorbar(y=range(len(coefficients_df)),
                   x=coefficients_df['Coefficient'],
                   xerr=[coefficients_df['Coefficient'] - coefficients_df['CI_Lower'],
                         coefficients_df['CI_Upper'] - coefficients_df['Coefficient']],
                   fmt='o', markersize=markersize, capsize=capsize, color='skyblue', ecolor='black')

# Plot asterisks for significant predictors (p-value < 0.05)
for i, p_value in enumerate(p_vals):
    if p_value < 0.001:
        idx = np.where(coefficients_df.index.values == i)[0]
        plt.text(coefficients_df['Coefficient'][i], idx+0.3, '***', fontsize=legend_fontsize+10, ha='center',va='top', color='red')

    elif p_value < 0.01:
        idx = np.where(coefficients_df.index.values == i)[0]
        plt.text(coefficients_df['Coefficient'][i], idx+0.3, '**', fontsize=legend_fontsize+10, ha='center', va='top',color='red')

    elif p_value < 0.05:
        idx = np.where(coefficients_df.index.values == i)[0]
        plt.text(coefficients_df['Coefficient'][i], idx+0.3, '*', fontsize=legend_fontsize+10, ha='center', va='top',color='red')

plt.yticks(range(len(coefficients_df)), [y_labels[predictors[i]] for i in coefficients_df.index],ha='right', fontsize=label_fontsize)
plt.ylim(bottom=-0.5, top=2.5)
plt.xticks(fontsize=label_fontsize)
plt.xlabel(r'$\beta$-coefficient', fontsize=label_fontsize)
plt.axvline(0, linestyle='--', color='gray', linewidth=1)
plt.text(0.8,1.9,f'$R^{2}$ = ' + f'{r_squared:.2f}', fontsize=title_fontsize, ha='right', va='top')
plt.title("(a)", fontsize=title_fontsize)

#--------------------------------- %GDP
ax = fig.add_subplot(gs[1, 0])
ax = sns.scatterplot(data=df_to_plot, x='y_actual_pct', y='y_predicted_pct', hue='region_ar6_6', palette=lab_colors, s=100, legend=False)

lim_min = 5e-3
lim_max = 5e1

# Plot the diagonal line where y_actual equals y_predicted
max_val = max(df_to_plot['y_actual_usd'].max(), df_to_plot['y_predicted_usd'].max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k-', linewidth=0.8)
#
# Set axis scales to log
plt.xscale('log')
plt.yscale('log')

# Set axis limits
plt.xlim(lim_min, lim_max)
plt.ylim(lim_min, lim_max)

# Adding labels, title, and legend
plt.xlabel(r"Reported damage [%GDP]",fontsize=label_fontsize)
plt.ylabel(r"Predicted damage [%GDP]",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.title("(b)", fontsize=title_fontsize)

# Add grid lines
plt.grid(True, which="major", ls="-", linewidth=0.5)

#--------------------------------- USD
ax = fig.add_subplot(gs[1, 1])
ax = sns.scatterplot(data=df_to_plot, x='y_actual_usd', y='y_predicted_usd', hue='region_ar6_6', palette=lab_colors, s=100)

lim_min = 1e6
lim_max = 1e12

# Plot the diagonal line where y_actual equals y_predicted
max_val = max(df_to_plot['y_actual_usd'].max(), df_to_plot['y_predicted_usd'].max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k-', linewidth=0.8)

# Set axis scales to log
plt.xscale('log')
plt.yscale('log')

# Set axis limits
plt.xlim(lim_min, lim_max)
plt.ylim(lim_min, lim_max)

# Define a formatter for scientific notation
formatter = FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$')

# Apply the formatter to both x and y axes
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

# Set grid lines only at 10^x intervals
plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))

# Adding labels, title, and legend
plt.xlabel(f"Reported damage [US$]",fontsize=label_fontsize)
plt.ylabel(f"Predicted damage [US$]",fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.title("(c)", fontsize=title_fontsize)

# Add grid lines
plt.grid(True, which="major", ls="-", linewidth=0.5)

# LEGEND
handles, labels = plt.gca().get_legend_handles_labels()
labels = [legend_labs.get(label, label) for label in labels]
label_handle_map = dict(zip(labels, handles))
sorted_labels = sorted(labels) # sort alphabetically
sorted_handles = [label_handle_map[label] for label in sorted_labels]
ax.legend(sorted_handles, sorted_labels, loc='lower center', ncol=2, bbox_to_anchor=(-0.3, -1.1), frameon=False, fontsize=legend_fontsize)

# -----------------------------------------
fig.subplots_adjust(left=0.18,
                    bottom=0.28,
                    right=0.95,
                    top=0.94,
                    wspace=0.5,  # 0.02
                    hspace=0.65)  # 0.65

"""
wspace and hspace specify the space reserved between Matplotlib subplots. They are the fractions of axis width and height, respectively.
left, right, top and bottom parameters specify four sides of the subplots’ positions. They are the fractions of the width and height of the figure.
top and bottom should add up to 1.0
"""

plt.show()